"""Visualization utilities for crack segmentation results."""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def save_prediction_grid(images, pred_masks, gt_masks, save_path,
                         num_samples=8, threshold=0.5):
    """Save a grid of input / prediction / ground truth comparisons.

    Args:
        images: (N, 3, H, W) or list
        pred_masks: (N, 1, H, W) logits or probabilities
        gt_masks: (N, 1, H, W) binary
        save_path: output file path
        num_samples: number of samples to show
        threshold: binarization threshold
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().numpy()
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.cpu().numpy()

    num_samples = min(num_samples, len(images))
    ncols = 3  # Input | Prediction | GT
    nrows = num_samples

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Input image
        img = images[i].transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        # Prediction
        pred = pred_masks[i, 0]
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred_prob = 1.0 / (1.0 + np.exp(-pred))
        else:
            pred_prob = pred
        pred_bin = (pred_prob > threshold).astype(np.float32)
        axes[i, 1].imshow(pred_bin, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Prediction (IoU: ...)")
        axes[i, 1].axis("off")

        # Ground truth
        gt = gt_masks[i, 0]
        axes[i, 2].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_curves(train_metrics, val_metrics, save_path):
    """Plot training and validation metric curves.

    Args:
        train_metrics: list of dicts per epoch
        val_metrics: list of dicts per epoch
        save_path: output file path
    """
    epochs = range(1, len(train_metrics) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curve
    if "loss" in train_metrics[0]:
        axes[0].plot(epochs, [m["loss"] for m in train_metrics],
                     label="Train Loss", marker=".")
        if val_metrics and "loss" in val_metrics[0]:
            axes[0].plot(epochs, [m["loss"] for m in val_metrics],
                         label="Val Loss", marker=".")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # IoU curve
    if "iou" in train_metrics[0]:
        axes[1].plot(epochs, [m["iou"] for m in train_metrics],
                     label="Train IoU", marker=".")
        if val_metrics and "iou" in val_metrics[0]:
            axes[1].plot(epochs, [m["iou"] for m in val_metrics],
                         label="Val IoU", marker=".")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("IoU")
        axes[1].set_title("IoU")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # F1 curve
    if "f1" in train_metrics[0]:
        axes[2].plot(epochs, [m["f1"] for m in train_metrics],
                     label="Train F1", marker=".")
        if val_metrics and "f1" in val_metrics[0]:
            axes[2].plot(epochs, [m["f1"] for m in val_metrics],
                         label="Val F1", marker=".")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("F1")
        axes[2].set_title("F1 Score")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()