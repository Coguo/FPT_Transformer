"""Evaluation metrics for crack segmentation.

Implements three metrics from the paper:
  1. IoU (Intersection over Union)
  2. F1 Score (Dice coefficient)
  3. BF (Boundary F1) - boundary-aware F1 score
"""

import torch
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def compute_iou(pred: torch.Tensor, target: torch.Tensor,
                smooth: float = 1e-6) -> torch.Tensor:
    """Intersection over Union.

    Args:
        pred: (B, 1, H, W) binary {0, 1}
        target: (B, 1, H, W) binary {0, 1}
        smooth: smoothing factor to avoid div by zero
    Returns:
        (B,) per-sample IoU
    """
    pred = pred.contiguous().view(pred.shape[0], -1).float()
    target = target.contiguous().view(target.shape[0], -1).float()

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou


def compute_f1(pred: torch.Tensor, target: torch.Tensor,
               smooth: float = 1e-6) -> torch.Tensor:
    """F1 Score (Dice coefficient).

    F1 = 2 * TP / (2 * TP + FP + FN)

    Args:
        pred: (B, 1, H, W) binary {0, 1}
        target: (B, 1, H, W) binary {0, 1}
    Returns:
        (B,) per-sample F1
    """
    pred = pred.contiguous().view(pred.shape[0], -1).float()
    target = target.contiguous().view(target.shape[0], -1).float()

    tp = (pred * target).sum(dim=1)
    fp = pred.sum(dim=1) - tp
    fn = target.sum(dim=1) - tp

    f1 = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    return f1


def compute_bf(pred: torch.Tensor, target: torch.Tensor,
               dilation_radius: int = 2) -> torch.Tensor:
    """Boundary F1 score.

    Extracts boundaries via morphological gradient, dilates them,
    then computes F1 on the boundary regions.

    Args:
        pred: (B, 1, H, W) binary {0, 1}
        target: (B, 1, H, W) binary {0, 1}
        dilation_radius: radius for boundary dilation (default 2)
    Returns:
        (B,) per-sample BF score
    """
    B = pred.shape[0]
    bf_scores = []

    for i in range(B):
        p = pred[i, 0].cpu().numpy().astype(bool)
        t = target[i, 0].cpu().numpy().astype(bool)

        # Extract boundaries: morphological gradient = dilation - erosion
        p_boundary = binary_dilation(p, iterations=1) & ~binary_erosion(
            p, iterations=1
        )
        t_boundary = binary_dilation(t, iterations=1) & ~binary_erosion(
            t, iterations=1
        )

        # Dilate boundaries for tolerance
        p_boundary = binary_dilation(p_boundary, iterations=dilation_radius)
        t_boundary = binary_dilation(t_boundary, iterations=dilation_radius)

        # Precision and recall on boundary pixels
        tp = (p_boundary & t_boundary).sum()
        fp = (p_boundary & ~t_boundary).sum()
        fn = (~p_boundary & t_boundary).sum()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        if precision + recall > 0:
            bf = 2 * precision * recall / (precision + recall)
        else:
            bf = 0.0

        bf_scores.append(bf)

    return torch.tensor(bf_scores, device=pred.device)


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor,
                        threshold: float = 0.5,
                        compute_bf_flag: bool = True) -> dict:
    """Compute all metrics at once.

    Args:
        pred: (B, 1, H, W) logits or probabilities
        target: (B, 1, H, W) binary {0, 1}
        threshold: threshold for binarization
        compute_bf_flag: whether to compute BF (slower)
    Returns:
        dict with 'iou', 'f1', 'bf' (per-sample as list or None)
    """
    # Binarize
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred_bin = (torch.sigmoid(pred) > threshold).float()
    else:
        pred_bin = (pred > threshold).float()

    iou = compute_iou(pred_bin, target)
    f1 = compute_f1(pred_bin, target)

    result = {
        "iou": iou.mean().item(),
        "f1": f1.mean().item(),
        "iou_per_sample": iou.tolist(),
        "f1_per_sample": f1.tolist(),
    }

    if compute_bf_flag:
        bf = compute_bf(pred_bin, target)
        result["bf"] = bf.mean().item()
        result["bf_per_sample"] = bf.tolist()
    else:
        result["bf"] = 0.0

    return result