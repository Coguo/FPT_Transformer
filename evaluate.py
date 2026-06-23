"""Evaluation pipeline for FPT model.

Computes all metrics (IoU, F1, BF) on the test set,
saves visualizations, and generates a summary report.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Ensure the parent directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import FPTNet
from utils.metrics import compute_all_metrics
from utils.visualization import save_prediction_grid


class Evaluator:
    """Full evaluation pipeline."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, test_loader, config, save_dir="./eval_results",
                 threshold=0.5):
        """Run full evaluation.

        Args:
            test_loader: DataLoader for test set
            config: configuration dict
            save_dir: directory to save results
            threshold: binarization threshold

        Returns:
            dict with averaged metrics
        """
        self.model.eval()
        os.makedirs(save_dir, exist_ok=True)

        all_preds = []
        all_masks = []
        all_images = []

        pbar = tqdm(test_loader, desc="Evaluating")
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            logits = self.model(images)

            all_preds.append(logits.cpu())
            all_masks.append(masks.cpu())
            all_images.append(images.cpu())

        # Concatenate
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        all_images = torch.cat(all_images, dim=0)

        # Compute metrics
        metrics = compute_all_metrics(
            all_preds, all_masks,
            threshold=threshold,
            compute_bf_flag=True,
        )

        # Per-image stats
        iou_per_sample = np.array(metrics["iou_per_sample"])
        f1_per_sample = np.array(metrics["f1_per_sample"])

        print(f"\n{'='*60}")
        print(f"Evaluation Results")
        print(f"  Samples: {len(all_preds)}")
        print(f"  Threshold: {threshold}")
        print(f"{'='*60}")
        print(f"  IoU: {metrics['iou']:.4f} +/- {iou_per_sample.std():.4f}")
        print(f"  F1:  {metrics['f1']:.4f} +/- {f1_per_sample.std():.4f}")
        if "bf" in metrics:
            bf_per_sample = np.array(metrics.get("bf_per_sample", [0]))
            print(f"  BF:  {metrics['bf']:.4f} +/- {bf_per_sample.std():.4f}")
        print(f"{'='*60}\n")

        # Save visualization grid
        vis_path = os.path.join(save_dir, "predictions.png")
        save_prediction_grid(
            all_images, all_preds, all_masks, vis_path,
            num_samples=min(
                config.get("eval", {}).get("max_vis_samples", 16),
                len(all_images),
            ),
            threshold=threshold,
        )
        print(f"Visualization saved: {vis_path}")

        return metrics