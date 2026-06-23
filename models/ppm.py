"""Pyramid Pooling Module (PPM).

Adapted from PSPNet (Zhao et al., 2017).
Used in FPT Stage 5 to pool the coarsest backbone feature
at multiple scales for rich contextual information.

Input:  (B, C_in, H, W)
Output: (B, out_channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    """PPM with configurable pool scales.

    For each scale, applies AdaptiveAvgPool -> 1x1 Conv -> BN -> ReLU
    -> upsample back to input resolution, then concatenates all
    pooled features with the original (via reduction conv) and
    applies a bottleneck convolution.
    """

    def __init__(self, in_channels, out_channels=96,
                 pool_scales=(1, 2, 3, 6), inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = in_channels // 4  # 768//4 = 192

        self.branches = nn.ModuleList()
        for scale in pool_scales:
            self.branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, inter_channels, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
            ))

        # Reduction conv for the original feature (before pooling)
        self.reduction = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # After concat: total channels = out_channels + n_branches * inter_channels
        concat_channels = out_channels + len(pool_scales) * inter_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = out_channels

    def forward(self, x):
        """Args:
            x: (B, C_in, H, W)
        Returns:
            (B, out_channels, H, W)
        """
        H, W = x.shape[2:]

        # Branch features: each branch pools then upsamples back
        branch_out = []
        for branch in self.branches:
            pooled = branch(x)          # (B, inter, s, s)
            upsampled = F.interpolate(
                pooled, size=(H, W),
                mode="bilinear", align_corners=False,
            )
            branch_out.append(upsampled)

        # Original feature through reduction
        reduced = self.reduction(x)     # (B, out_channels, H, W)

        # Concatenate and bottleneck
        concat = torch.cat([reduced] + branch_out, dim=1)
        out = self.bottleneck(concat)

        return out
