"""FPT: Feature Pyramid Transformer for Concrete Crack Segmentation.

Entry point for training, evaluation, and inference.

Usage:
    python main.py --mode train --config configs/default.yaml
    python main.py --mode eval --checkpoint checkpoints/best_fpt.pth
    python main.py --mode demo --image path/to/image.jpg --checkpoint checkpoints/best_fpt.pth
"""

import os
import sys
import argparse
import yaml

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import FPTNet
from data import get_dataloaders
from trainer import Trainer
from evaluate import Evaluator


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_mode(config):
    """Run full training pipeline."""
    print("=" * 60)
    print("FPT: Feature Pyramid Transformer for Crack Segmentation")
    print("=" * 60)

    # Data
    print("\n[1/3] Preparing data...")
    train_loader, val_loader = get_dataloaders(config)
    print(f"  Train samples: {config['data']['num_train']}")
    print(f"  Test samples:  {config['data']['num_test']}")
    print(f"  Batch size:    {config['data']['batch_size']}")
    print(f"  Image size:    {config['data']['input_size']}x{config['data']['input_size']}")

    # Trainer
    print("\n[2/3] Initializing model...")
    trainer = Trainer(config)

    # Phase 1: FPT Training
    print("\n[3/3] Training...")
    trainer.train_fpt_phase(train_loader, val_loader)

    # Phase 2: IBR Fine-tuning (optional)
    if config["training"]["ibr"]["enabled"]:
        # Need direction data for IBR
        train_loader_ibr, val_loader_ibr = get_dataloaders(config)
        trainer.train_ibr_phase(train_loader_ibr, val_loader_ibr)

    trainer.close()
    print("\nTraining complete!")


def eval_mode(config, checkpoint_path):
    """Run evaluation on test set."""
    print(f"Loading checkpoint: {checkpoint_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Model
    model = FPTNet(
        num_classes=1,
        pretrained=False,
        decoder_type=config["model"]["decoder"]["type"],
    ).to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded epoch {checkpoint['epoch']} with "
          f"F1={checkpoint['metrics'].get('f1', 0.0):.4f}")

    # Data
    _, test_loader = get_dataloaders(config)

    # Evaluate
    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(
        test_loader, config,
        save_dir=os.path.join(
            os.path.dirname(checkpoint_path), "eval_results"
        ),
        threshold=config["eval"]["threshold"],
    )

    return metrics


def demo_mode(config, checkpoint_path, image_path):
    """Run inference on a single image."""
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Model
    model = FPTNet(
        num_classes=1,
        pretrained=False,
        decoder_type=config["model"]["decoder"]["type"],
    ).to(device)
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    input_size = config["data"]["input_size"]
    img_resized = img.resize((input_size, input_size), Image.BILINEAR)

    img_tensor = torch.from_numpy(
        np.array(img_resized).transpose(2, 0, 1)
    ).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        pred = torch.sigmoid(logits)
        pred_bin = (pred > config["eval"]["threshold"]).float()

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_resized)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(pred[0, 0].cpu().numpy(), cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Prediction (probability)")
    axes[1].axis("off")

    axes[2].imshow(pred_bin[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (threshold={config['eval']['threshold']})")
    axes[2].axis("off")

    plt.tight_layout()

    # Save
    output_path = os.path.splitext(image_path)[0] + "_prediction.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Prediction saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="FPT: Feature Pyramid Transformer for Crack Segmentation"
    )
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "eval", "demo"],
        help="Operation mode",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (for eval/demo modes)",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to image file (for demo mode)",
    )
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = load_config(config_path)

    # Set random seeds
    torch.manual_seed(config["training"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["training"]["seed"])

    if args.mode == "train":
        train_mode(config)
    elif args.mode == "eval":
        if args.checkpoint is None:
            print("Error: --checkpoint is required for eval mode")
            sys.exit(1)
        eval_mode(config, args.checkpoint)
    elif args.mode == "demo":
        if args.checkpoint is None or args.image is None:
            print("Error: --checkpoint and --image are required for demo mode")
            sys.exit(1)
        demo_mode(config, args.checkpoint, args.image)


if __name__ == "__main__":
    main()