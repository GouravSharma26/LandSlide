"""Multimodal Landslide Segmentation Architecture integrating SAR, Optical, and DEM.

Combines independent modality encoders, availability-conditioned multi-scale fusion,
and a unified spatial decoder for reliable landslide segmentation under missing modalities.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geotrustnet.models.unet import Up, OutConv
from src.geotrustnet.models.encoders import SAREncoder, OpticalEncoder, TerrainEncoder
from src.geotrustnet.models.fusion import FeatureLevelFusion


class MultimodalUNet(nn.Module):
    """Multimodal U-Net for Sentinel-1 SAR, Sentinel-2 Optical, and Copernicus DEM Fusion.

    Args:
        features: Channel widths for encoder and decoder stages (default [32, 64, 128, 256]).
        sar_channels: Channels for SAR input (default 2: VV, VH).
        optical_channels: Channels for optical input (default 10: S2 spectral bands).
        dem_channels: Channels for DEM input (default 2: elevation, slope).
        out_channels: Output logits channels (default 1 for binary landslide mask).
        bilinear: Whether to use bilinear upsampling in decoder (default True).
        dropout: Spatial dropout rate in encoders and decoder.
        fusion_type: 'concat', 'gated', or 'residual_sar' (default 'concat').
    """

    def __init__(
        self,
        features: Optional[List[int]] = None,
        sar_channels: int = 2,
        optical_channels: int = 10,
        dem_channels: int = 2,
        out_channels: int = 1,
        bilinear: bool = True,
        dropout: float = 0.1,
        fusion_type: str = "concat",
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.features = list(features)
        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.dem_channels = dem_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.dropout = dropout
        self.fusion_type = fusion_type

        # 1. Independent Modality Encoders
        self.sar_encoder = SAREncoder(in_channels=sar_channels, features=features, dropout=dropout)
        self.optical_encoder = OpticalEncoder(in_channels=optical_channels, features=features, dropout=dropout)
        self.dem_encoder = TerrainEncoder(in_channels=dem_channels, features=features, dropout=dropout)

        # 2. Multi-Scale Feature Fusion Modules
        # Fuses intermediate skip connections at scales s0, s1, s2
        self.skip_fusions = nn.ModuleList()
        for f_dim in features[:-1]:
            self.skip_fusions.append(
                FeatureLevelFusion(
                    feature_dim=f_dim,
                    num_modalities=3,
                    out_dim=f_dim,
                    dropout=dropout,
                    fusion_type=fusion_type,
                )
            )

        # Fuses deepest bottleneck representations
        self.bottleneck_fusion = FeatureLevelFusion(
            feature_dim=features[-1],
            num_modalities=3,
            out_dim=features[-1],
            dropout=dropout,
            fusion_type=fusion_type,
        )

        # 3. Spatial Decoder with Skip Connections
        self.ups = nn.ModuleList()
        rev_features = list(reversed(features))
        for i in range(len(rev_features) - 1):
            curr_in = rev_features[i]
            skip_in = rev_features[i + 1]
            out_ch = rev_features[i + 1]
            self.ups.append(
                Up(
                    in_channels=curr_in,
                    skip_channels=skip_in,
                    out_channels=out_ch,
                    bilinear=bilinear,
                    dropout=dropout,
                )
            )

        # 4. Final Projection to Logits
        self.outc = OutConv(features[0], out_channels)

    def _prepare_modality_input(
        self,
        tensor: Optional[torch.Tensor],
        expected_channels: int,
        ref_shape: Tuple[int, int, int],
        device: torch.device,
        has_flag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate, reshape, or instantiate zero fallback for a modality input."""
        B, H, W = ref_shape

        if tensor is None:
            x = torch.zeros((B, expected_channels, H, W), device=device, dtype=torch.float32)
            avail = torch.zeros(B, device=device, dtype=torch.bool)
            return x, avail

        # If 5D temporal tensor [B, T, C, H, W], take latest time slice
        if tensor.ndim == 5:
            tensor = tensor[:, -1, :, :, :]

        x = tensor.to(device).float()

        if has_flag is not None:
            avail = has_flag.to(device).bool()
            if avail.ndim > 1:
                avail = avail.squeeze()
        else:
            avail = torch.ones(B, device=device, dtype=torch.bool)

        return x, avail

    def forward(
        self,
        sar: Optional[torch.Tensor] = None,
        optical: Optional[torch.Tensor] = None,
        dem: Optional[torch.Tensor] = None,
        has_sar: Optional[torch.Tensor] = None,
        has_optical: Optional[torch.Tensor] = None,
        has_dem: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multimodal forward pass with missing-modality handling.

        Args:
            sar: Sentinel-1 SAR [B, 2, H, W] or [B, T, 2, H, W] (optional)
            optical: Sentinel-2 Optical [B, 10, H, W] or [B, T, 10, H, W] (optional)
            dem: Copernicus DEM [B, 2, H, W] (optional)
            has_sar: Boolean indicator [B] for SAR validity
            has_optical: Boolean indicator [B] for Optical validity
            has_dem: Boolean indicator [B] for DEM validity

        Returns:
            logits: Predicted class logits [B, out_channels, H, W]
        """
        # Determine reference batch size and spatial dimensions from any available tensor
        ref_tensor = sar if sar is not None else (optical if optical is not None else dem)
        if ref_tensor is None:
            raise ValueError("At least one modality tensor (sar, optical, or dem) must be provided.")

        B = ref_tensor.size(0)
        H, W = ref_tensor.size(-2), ref_tensor.size(-1)
        device = ref_tensor.device
        ref_shape = (B, H, W)

        # Prepare and validate all 3 modalities
        x_sar, a_sar = self._prepare_modality_input(sar, self.sar_channels, ref_shape, device, has_sar)
        x_opt, a_opt = self._prepare_modality_input(optical, self.optical_channels, ref_shape, device, has_optical)
        x_dem, a_dem = self._prepare_modality_input(dem, self.dem_channels, ref_shape, device, has_dem)

        avails = [a_sar, a_opt, a_dem]

        # 1. Hierarchical Feature Extraction per Modality
        bot_sar, skips_sar = self.sar_encoder(x_sar, is_available=a_sar)
        bot_opt, skips_opt = self.optical_encoder(x_opt, is_available=a_opt)
        bot_dem, skips_dem = self.dem_encoder(x_dem, is_available=a_dem)

        # 2. Multi-Scale Feature Fusion
        # Fuse intermediate skips at each scale (s0, s1, s2)
        fused_skips = []
        for i, skip_fusion_module in enumerate(self.skip_fusions):
            scale_feats = [skips_sar[i], skips_opt[i], skips_dem[i]]
            fused_s = skip_fusion_module(scale_feats, avails)
            fused_skips.append(fused_s)

        # Fuse deepest bottleneck
        bot_feats = [bot_sar, bot_opt, bot_dem]
        fused_bottleneck = self.bottleneck_fusion(bot_feats, avails)

        # 3. Spatial Decoder with Fused Skip Connections
        cur = fused_bottleneck
        for i, up in enumerate(self.ups):
            skip = fused_skips[-(i + 1)]
            cur = up(cur, skip)

        logits = self.outc(cur)
        return logits

    def predict_proba(
        self,
        sar: Optional[torch.Tensor] = None,
        optical: Optional[torch.Tensor] = None,
        dem: Optional[torch.Tensor] = None,
        has_sar: Optional[torch.Tensor] = None,
        has_optical: Optional[torch.Tensor] = None,
        has_dem: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run forward pass and return predicted probabilities in [0, 1]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                sar=sar,
                optical=optical,
                dem=dem,
                has_sar=has_sar,
                has_optical=has_optical,
                has_dem=has_dem,
            )
            return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        """Return total count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
