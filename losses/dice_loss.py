"""Dice Loss implementation for binary segmentation.

L_dice = 1 - (2 * sum(p * g) + smooth) / (sum(p) + sum(g) + smooth)

Used as the primary loss for FPT training (paper: 300 epochs with Dice loss).
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice coefficient loss for binary segmentation.

    Supports:
    - Standard Dice: (2 * |P ∩ G|) / (|P| + |G|)
    - Squared Dice: (2 * sum(p^2 * g^2)) / (sum(p^2) + sum(g^2))
    - Multi-class via channel dimension
    """

    def __init__(self, smooth=1e-5, squared=False, reduction="mean"):
        super().__init__()
        self.smooth = smooth
        self.squared = squared
        self.reduction = reduction

    def forward(self, pred_logits, target):
        """
        Args:
            pred_logits: (B, 1, H, W) or (B, C, H, W) raw logits
            target: (B, 1, H, W) binary {0, 1} or (B, C, H, W) one-hot
        Returns:
            Dice loss (scalar)
        """
        # Apply sigmoid
        pred = torch.sigmoid(pred_logits)

        # Flatten spatial dimensions
        B, C, H, W = pred.shape
        pred = pred.contiguous().view(B, C, -1)
        target = target.contiguous().view(B, C, -1)

        # Reduce across spatial dimensions
        if self.squared:
            pred = pred ** 2
            target = target ** 2

        intersection = (pred * target).sum(dim=2)  # (B, C)
        pred_sum = pred.sum(dim=2)
        target_sum = target.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (
            pred_sum + target_sum + self.smooth
        )  # (B, C)

        loss = 1.0 - dice  # (B, C)

        # Reduce
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction is None:
            pass  # return per-class per-batch

        return loss