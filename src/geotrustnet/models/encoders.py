"""Modality-specific feature encoders for multimodal Earth Observation data.

Provides independent 2D CNN encoders for Sentinel-1 SAR, Sentinel-2 Optical,
and Copernicus DEM terrain data with explicit modality availability masking.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from src.geotrustnet.models.unet import DoubleConv, Down


class ModalityEncoder(nn.Module):
    """Generic multi-scale 2D CNN feature encoder for a single EO modality.

    Args:
        in_channels: Number of raw input channels (e.g., 2 for SAR, 10 for Optical, 2 for DEM).
        features: Channel widths for hierarchical stages (default [32, 64, 128, 256]).
        dropout: Spatial dropout rate between convolution layers.
        modality_name: Identifier string for debugging and metadata.
    """

    def __init__(
        self,
        in_channels: int,
        features: Optional[List[int]] = None,
        dropout: float = 0.1,
        modality_name: str = "generic",
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.features = list(features)
        self.dropout = dropout
        self.modality_name = modality_name

        # Stem
        self.inc = DoubleConv(in_channels, features[0], dropout=dropout)

        # Downscaling stages
        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(
                Down(features[i], features[i + 1], dropout=dropout)
            )

    def forward(
        self,
        x: torch.Tensor,
        is_available: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through hierarchical encoder stages.

        Args:
            x: Input tensor [B, in_channels, H, W]
            is_available: Optional boolean/binary availability mask [B] or [B, 1, 1, 1].
                          If False/0 for a sample, all output features are zeroed out.

        Returns:
            bottleneck: Deepest feature map [B, features[-1], H_bot, W_bot]
            skips: List of intermediate multi-scale skip features [s0, s1, s2]
        """
        batch_size = x.size(0)

        # Multi-scale feature extraction
        s0 = self.inc(x)
        skips = [s0]

        cur = s0
        for down in self.downs:
            cur = down(cur)
            skips.append(cur)

        bottleneck = skips[-1]
        intermediate_skips = skips[:-1]

        # Apply modality availability mask if provided
        if is_available is not None:
            if is_available.ndim == 1:
                mask_4d = is_available.view(batch_size, 1, 1, 1).float()
            else:
                mask_4d = is_available.float()

            bottleneck = bottleneck * mask_4d
            intermediate_skips = [s * mask_4d for s in intermediate_skips]

        return bottleneck, intermediate_skips


class SAREncoder(ModalityEncoder):
    """Specialized encoder for Sentinel-1 SAR (VV, VH backscatter in dB)."""

    def __init__(
        self,
        in_channels: int = 2,
        features: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_channels=in_channels,
            features=features,
            dropout=dropout,
            modality_name="sar",
        )


class OpticalEncoder(ModalityEncoder):
    """Specialized encoder for Sentinel-2 MSI Optical Imagery (10 spectral bands)."""

    def __init__(
        self,
        in_channels: int = 10,
        features: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_channels=in_channels,
            features=features,
            dropout=dropout,
            modality_name="optical",
        )


class TerrainEncoder(ModalityEncoder):
    """Specialized encoder for Copernicus DEM Terrain features (Elevation and Slope)."""

    def __init__(
        self,
        in_channels: int = 2,
        features: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_channels=in_channels,
            features=features,
            dropout=dropout,
            modality_name="dem",
        )
