"""
Swin-T Backbone Wrapper

Wraps timm's Swin-Tiny (patch4, window7) with features_only=True,
converts NHWC feature maps to NCHW, and handles input resizing
from the original image size to Swin-T's native 224x224 resolution.

Output: list of 4 feature maps at stages 1-4
  Stage 1: (B,  96, H/4,  W/4)
  Stage 2: (B, 192, H/8,  W/8)
  Stage 3: (B, 384, H/16, W/16)
  Stage 4: (B, 768, H/32, W/32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SwinTBackbone(nn.Module):
    """Swin-Tiny feature extractor with NCHW output."""

    def __init__(self, backbone_name="swin_tiny_patch4_window7_224",
                 pretrained=True, out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.model = timm.create_model(
            backbone_name,
            features_only=True,
            pretrained=pretrained,
            out_indices=out_indices,
        )
        # Model expects NHWC internally; we'll convert output to NCHW
        self.model.eval() if not pretrained else None  # train mode for training
        self.channels = self.model.feature_info.channels()
        # [96, 192, 384, 768] for Swin-Tiny
        self._backbone_input_size = 224  # Swin-T native size
        self._padded_size = 224  # 224 // 32 = 7, divisible by window7

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image at original resolution (e.g., 320)
        Returns:
            list of 4 NCHW feature maps
        """
        B, C, H, W = x.shape

        # Resize to backbone's native input size if needed
        if H != self._backbone_input_size or W != self._backbone_input_size:
            x = F.interpolate(
                x, size=(self._backbone_input_size, self._backbone_input_size),
                mode="bilinear", align_corners=False,
            )

        # Forward through Swin-T: returns NHWC feature maps
        feats = self.model(x)  # list of NHWC tensors

        # Convert NHWC -> NCHW
        nchw_feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]

        return nchw_feats  # list of 4 tensors


def get_backbone_channels():
    """Return channel dimensions for each stage of Swin-Tiny."""
    return [96, 192, 384, 768]
