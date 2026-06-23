"""FPTNet: Complete Feature Pyramid Transformer segmentation model.

Assembles the full pipeline:
  1. Swin-T Backbone (4-stage feature extractor)
  2. FPT Neck (Stage 5: PPM + feature fusion)
  3. Segmentation Decoder (Simple or IBR)

Input:  (B, 3, H, W)  -- any resolution (internally resized to 224 for backbone)
Output: (B, 1, H, W)  -- segmentation logits at original resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_backbone import SwinTBackbone
from .fpt_neck import FPTNeck
from .decoder import SimpleDecoder, IBRDecoder


class FPTNet(nn.Module):
    """Complete FPT model for crack segmentation.

    Architecture (matching the paper):
      Backbone: Swin-Tiny (patch4, window7)
      Neck:     FPT Neck (PPM + feature fusion to 96ch at 1/4 scale)
      Head:     Simple decoder or IBR decoder (boundary+direction refinement)
    """

    def __init__(self, num_classes=1, pretrained=True, decoder_type="simple",
                 neck_channels=96):
        super().__init__()

        self.backbone = SwinTBackbone(pretrained=pretrained)
        self.neck = FPTNeck(
            stage3_channels=384,
            stage4_channels=768,
            neck_channels=neck_channels,
            ppm_scales=(1, 2, 3, 6),
        )

        if decoder_type == "simple":
            self.decoder = SimpleDecoder(
                in_channels=neck_channels,
                hidden_channels=(64, 32),
                num_classes=num_classes,
            )
        elif decoder_type == "ibr":
            self.decoder = IBRDecoder(
                in_channels=neck_channels,
                hidden_dim=64,
                num_classes=num_classes,
            )
        else:
            raise ValueError(
                f"Unknown decoder type: {decoder_type}. "
                "Choose 'simple' or 'ibr'."
            )

        self.decoder_type = decoder_type
        self.input_size = None  # Will be set on first forward

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image

        Returns:
            For simple decoder:
                (B, num_classes, H, W) segmentation logits
            For IBR decoder:
                (logits, boundary, direction) tuple
        """
        B, C, H, W = x.shape
        self.input_size = (H, W)

        # 1. Swin-T Backbone (internally resizes to 224)
        feats = self.backbone(x)   # list of 4 NCHW tensors
        # feats[0]: (B, 96, 56, 56)
        # feats[1]: (B, 192, 28, 28)
        # feats[2]: (B, 384, 14, 14)  <- Stage 3
        # feats[3]: (B, 768, 7, 7)    <- Stage 4

        # 2. FPT Neck (Stage 5)
        neck_out = self.neck(feats[2], feats[3])
        # (B, 96, H/4*, W/4*)  -- at 224 scale: (B, 96, 56, 56)

        # 3. Decoder head
        out = self.decoder(neck_out)

        # 4. Resize to original input resolution
        if self.decoder_type == "simple":
            out = F.interpolate(out, size=(H, W), mode="bilinear",
                                align_corners=False)
            # (B, num_classes, H, W) at original resolution
        elif self.decoder_type == "ibr":
            logits, boundary, direction = out
            logits = F.interpolate(logits, size=(H, W), mode="bilinear",
                                   align_corners=False)
            boundary = F.interpolate(boundary, size=(H, W), mode="bilinear",
                                     align_corners=False)
            direction = F.interpolate(direction, size=(H, W), mode="bilinear",
                                      align_corners=False)
            out = (logits, boundary, direction)

        return out

    def get_trainable_params(self, phase="fpt"):
        """Get parameters for different training phases.

        Args:
            phase: "fpt" (train all) or "ibr" (train decoder only)
        Returns:
            list of parameters to optimize
        """
        if phase == "fpt":
            return self.parameters()
        elif phase == "ibr":
            # Only train decoder (head), freeze backbone + neck
            return self.decoder.parameters()
        else:
            raise ValueError(f"Unknown phase: {phase}")