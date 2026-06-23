"""Combined loss functions for FPT training.

Supports:
  1. FPTLoss: Dice loss for FPT phase training
  2. IBRLoss: BCE + CE combined loss for IBR fine-tuning phase
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice_loss import DiceLoss


class FPTLoss(nn.Module):
    """Composite loss for FPT training phase.

    Primary loss: Dice (recommended by the paper)
    Optional: Dice + BCE for improved convergence
    """

    def __init__(self, dice_weight=1.0, bce_weight=0.0, smooth=1e-5):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred_logits, target):
        """
        Args:
            pred_logits: (B, 1, H, W) raw logits
            target: (B, 1, H, W) binary {0, 1}
        Returns:
            Combined loss scalar
        """
        dice_loss = self.dice(pred_logits, target)
        bce_loss = self.bce(pred_logits, target)

        total = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return total


class IBRLoss(nn.Module):
    """Loss for IBR fine-tuning phase.

    L = Dice(seg_pred, seg_gt)
      + BCE(boundary_pred, boundary_gt)
      + CE(direction_pred, direction_gt)

    As described in the paper: BCE for boundary, CrossEntropy for direction.
    Note: the direction values are 8-way classification (0-7 directions).
    """

    def __init__(self, bce_weight=1.0, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.ce_weight = ce_weight

    def forward(self, pred, target):
        """
        Args:
            pred: tuple of (seg_logits, boundary_logits, direction_logits)
            target: tuple of (seg_gt, boundary_gt, direction_gt)
                seg_gt: (B, 1, H, W) binary
                boundary_gt: (B, 1, H, W) binary
                direction_gt: (B, H, W) long int class indices (0-7)
        Returns:
            dict with individual losses and total
        """
        seg_pred, boundary_pred, direction_pred = pred
        seg_gt, boundary_gt, direction_gt = target

        # Dice on segmentation
        seg_loss = self.dice_weight * self.dice(seg_pred, seg_gt)

        # BCE on boundary
        bce_loss = self.bce_weight * self.bce(boundary_pred, boundary_gt)

        # CE on direction (8-class)
        # direction_pred: (B, 8, H, W), direction_gt: (B, H, W)
        ce_loss = self.ce_weight * self.ce(direction_pred, direction_gt)

        total = seg_loss + bce_loss + ce_loss

        return {
            "total": total,
            "seg_loss": seg_loss.detach(),
            "bce_loss": bce_loss.detach(),
            "ce_loss": ce_loss.detach(),
        }