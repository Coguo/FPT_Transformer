"""Segmentation decoder heads for FPT.

Two options:
  1. SimpleDecoder: lightweight conv-based upsampling head
  2. IBRDecoder: Independent Boundary Refinement head (paper's full design)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDecoder(nn.Module):
    """Lightweight segmentation decoder.

    Takes the FPT neck output (96 channels at 1/4 scale) and
    upsamples to full-resolution segmentation logits.

    Architecture:
        Conv(96->64, 3) -> BN -> ReLU
        Upsample 2x
        Conv(64->32, 3) -> BN -> ReLU
        Upsample 2x
        Conv(32->num_classes, 1) -> logits
    """

    def __init__(self, in_channels=96, hidden_channels=(64, 32),
                 num_classes=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(hidden_channels[1], num_classes,
                                  kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, 96, H/4, W/4)
        Returns:
            (B, num_classes, H, W) logits at original input resolution
        """
        # First conv + upsample
        x = self.conv1(x)                        # (B, 64, H/4, W/4)
        x = F.interpolate(x, scale_factor=2,
                          mode="bilinear",
                          align_corners=False)   # (B, 64, H/2, W/2)

        # Second conv + upsample
        x = self.conv2(x)                        # (B, 32, H/2, W/2)
        x = F.interpolate(x, scale_factor=2,
                          mode="bilinear",
                          align_corners=False)   # (B, 32, H, W)

        # Final 1x1 conv to num_classes
        logits = self.conv_out(x)                # (B, num_classes, H, W)
        return logits


class IBRDecoder(nn.Module):
    """Independent Boundary Refinement decoder.

    Implements the paper's IBR head which produces:
    - Binary segmentation map (1 channel)
    - Boundary map (1 channel)
    - Direction map (8 channels, one per 45-degree bin)

    Architecture:
        Shared: 1x1 Conv(96->64) -> BN -> ReLU
        Boundary branch: 1x1 Conv(64->1)
        Direction branch: 1x1 Conv(64->8)

    Then combines boundary and direction cues to refine
    the segmentation boundary.
    """

    def __init__(self, in_channels=96, hidden_dim=64, num_classes=1):
        super().__init__()

        # Shared feature projection
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Segmentation branch (direct prediction)
        self.seg_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

        # Boundary branch: Crack-FPN [305,306] style
        self.boundary_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        # Direction branch: 8 directions (0-360°, 45° per bin)
        self.direction_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 8, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 96, H/4, W/4) FPT neck output
        Returns:
            logits: (B, 1, H, W) - segmentation logits
            boundary: (B, 1, H, W) - boundary logits
            direction: (B, 8, H, W) - direction logits
        """
        # Upsample to full resolution first
        B, C, H, W = x.shape
        x_full = F.interpolate(x, scale_factor=4, mode="bilinear",
                               align_corners=False)

        # Segmentation
        logits = self.seg_branch(x_full)

        # Shared features for boundary/direction
        shared = self.shared(x_full)  # (B, 64, H, W)
        boundary = self.boundary_branch(shared)
        direction = self.direction_branch(shared)

        return logits, boundary, direction
