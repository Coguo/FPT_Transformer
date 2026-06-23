"""FPT Neck (Stage 5).

Feature Pyramid Transformer neck that fuses multi-scale features
from the Swin-T backbone using PPM (Pyramid Pooling Module) and
element-wise addition-based feature fusion.

Inputs:
  - stage3_feat: (B, 384, H/16, W/16)   from Swin-T Stage 3
  - stage4_feat: (B, 768, H/32, W/32)   from Swin-T Stage 4

Output:
  - (B, 96, H/4, W/4) fused multi-scale feature

Architecture (matching the paper):
  1. Stage4 -> PPM -> 96 channels at H/32
  2. Stage3 -> 1x1 Conv(384->96) -> 96 channels at H/16
  3. Upsample PPM output 2x, element-wise add with processed Stage3
     -> 96 channels at H/16
  4. Upsample fused feature 4x -> 96 channels at H/4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ppm import PyramidPoolingModule


class FPTNeck(nn.Module):
    """FPT Stage 5: multi-scale feature fusion with PPM."""

    def __init__(self, stage3_channels=384, stage4_channels=768,
                 neck_channels=96, ppm_scales=(1, 2, 3, 6)):
        super().__init__()

        # PPM on the coarsest feature (Stage 4)
        self.ppm = PyramidPoolingModule(
            in_channels=stage4_channels,
            out_channels=neck_channels,
            pool_scales=ppm_scales,
        )

        # 1x1 conv on Stage 3 to match neck channels
        self.stage3_conv = nn.Sequential(
            nn.Conv2d(stage3_channels, neck_channels, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(neck_channels),
            nn.ReLU(inplace=True),
        )

        # Optional fusion refinement conv (applied after element-wise add)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(neck_channels, neck_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(neck_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = neck_channels

    def forward(self, stage3_feat, stage4_feat):
        """
        Args:
            stage3_feat: (B, 384, H/16, W/16)
            stage4_feat: (B, 768, H/32, W/32)
        Returns:
            (B, 96, H/4, W/4)
        """
        # 1. PPM on Stage 4
        ppm_out = self.ppm(stage4_feat)     # (B, 96, H/32, W/32)

        # 2. 1x1 conv on Stage 3
        skip_out = self.stage3_conv(stage3_feat)  # (B, 96, H/16, W/16)

        # 3. Upsample PPM output 2x, element-wise add with Stage3
        ppm_up = F.interpolate(
            ppm_out, size=stage3_feat.shape[2:],
            mode="bilinear", align_corners=False,
        )
        fused = ppm_up + skip_out
        fused = self.fusion_conv(fused)     # (B, 96, H/16, W/16)

        # 4. Upsample fused feature 4x to H/4 scale
        # stage3 is at H/16, so H/4 = H/16 * 4
        target_h = stage3_feat.shape[2] * 4
        target_w = stage3_feat.shape[3] * 4
        neck_out = F.interpolate(
            fused, size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        )                                   # (B, 96, H/4, W/4)

        return neck_out
