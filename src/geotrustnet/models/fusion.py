"""Multimodal feature fusion modules for Earth Observation landslide detection.

Provides feature-level fusion with explicit modality availability masking,
indicator conditioning, and multi-scale skip alignment.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geotrustnet.models.unet import DoubleConv


class FeatureLevelFusion(nn.Module):
    """Concatenation-based feature fusion with modality availability indicator conditioning.

    Merges representations from SAR, Optical, and Terrain while preventing missing modalities
    from corrupting the fused representation.

    Args:
        feature_dim: Number of feature channels per modality at this scale.
        num_modalities: Number of candidate modalities (default 3: SAR, Optical, DEM).
        out_dim: Output fused feature dimension (default matches feature_dim).
        dropout: Dropout rate in fusion layers.
        fusion_type: 'concat', 'gated', or 'residual_sar' (default 'concat').
    """

    def __init__(
        self,
        feature_dim: int,
        num_modalities: int = 3,
        out_dim: Optional[int] = None,
        dropout: float = 0.1,
        fusion_type: str = "concat",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.out_dim = out_dim if out_dim is not None else feature_dim
        self.fusion_type = fusion_type

        # Input channels: (num_modalities * feature_dim) + num_modalities indicator channels
        in_channels = num_modalities * feature_dim + num_modalities

        if fusion_type == "concat":
            self.fusion_block = DoubleConv(
                in_channels=in_channels,
                out_channels=self.out_dim,
                dropout=dropout,
            )
        elif fusion_type == "gated":
            # Per-modality spatial gating
            self.gate_conv = nn.Sequential(
                nn.Conv2d(in_channels, num_modalities, kernel_size=1),
                nn.Sigmoid(),
            )
            self.fusion_block = DoubleConv(
                in_channels=num_modalities * feature_dim,
                out_channels=self.out_dim,
                dropout=dropout,
            )
        elif fusion_type == "residual_sar":
            # SAR feature is base; other modalities provide residual refinement
            self.residual_block = DoubleConv(
                in_channels=(num_modalities - 1) * feature_dim + (num_modalities - 1),
                out_channels=self.out_dim,
                dropout=dropout,
            )
            self.sar_proj = nn.Conv2d(feature_dim, self.out_dim, kernel_size=1)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self,
        features: List[torch.Tensor],
        availabilities: List[torch.Tensor],
    ) -> torch.Tensor:
        """Fuse multi-modal features at a single scale.

        Args:
            features: List of feature tensors [z_sar, z_opt, z_dem], each [B, C, H, W]
            availabilities: List of availability masks [a_sar, a_opt, a_dem], each [B] or [B, 1, 1, 1]

        Returns:
            fused_tensor: [B, out_dim, H, W]
        """
        if len(features) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} features, got {len(features)}"
            )

        batch_size, _, height, width = features[0].shape
        device = features[0].device

        # Standardize availability masks to [B, 1, H, W]
        std_avails = []
        for a in availabilities:
            if a.ndim == 1:
                mask_4d = a.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width).float()
            else:
                mask_4d = a.float()
                if mask_4d.shape[-2:] != (height, width):
                    mask_4d = F.interpolate(mask_4d, size=(height, width), mode="nearest")
            std_avails.append(mask_4d.to(device))

        # 1. Mask features by availability
        masked_features = [f * m for f, m in zip(features, std_avails)]

        if self.fusion_type == "concat":
            # Concatenate features and availability indicator maps
            combined = torch.cat(masked_features + std_avails, dim=1)
            fused = self.fusion_block(combined)
            return fused

        elif self.fusion_type == "gated":
            # Compute availability-aware spatial gates
            combined_for_gates = torch.cat(masked_features + std_avails, dim=1)
            gates = self.gate_conv(combined_for_gates)  # [B, num_modalities, H, W]

            gated_features = []
            for i, feat in enumerate(masked_features):
                g_i = gates[:, i : i + 1, :, :]
                gated_features.append(feat * g_i)

            combined_gated = torch.cat(gated_features, dim=1)
            fused = self.fusion_block(combined_gated)
            return fused

        elif self.fusion_type == "residual_sar":
            # SAR is index 0
            z_sar = masked_features[0]
            m_sar = std_avails[0]
            base_sar = self.sar_proj(z_sar) * m_sar

            other_feats = masked_features[1:]
            other_avails = std_avails[1:]

            if other_feats:
                other_combined = torch.cat(other_feats + other_avails, dim=1)
                residual = self.residual_block(other_combined)
                fused = base_sar + residual
            else:
                fused = base_sar
            return fused

        raise NotImplementedError(f"Unsupported fusion type: {self.fusion_type}")
