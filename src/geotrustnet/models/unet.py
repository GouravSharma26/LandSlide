"""U-Net architecture for single-modality Sentinel-1 SAR baseline segmentation.

Lightweight encoder-decoder network with skip connections, batch normalization,
and configurable depth/widths for binary landslide segmentation.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool2d then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling, spatial padding adjustment, skip-connection concatenation, and DoubleConv."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            total_cat_channels = in_channels + skip_channels
            self.conv = DoubleConv(total_cat_channels, out_channels, dropout=dropout)
        else:
            reduced_channels = max(1, in_channels // 2)
            self.up = nn.ConvTranspose2d(
                in_channels, reduced_channels, kernel_size=2, stride=2
            )
            total_cat_channels = reduced_channels + skip_channels
            self.conv = DoubleConv(total_cat_channels, out_channels, dropout=dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Upsample x1 and concatenate with skip connection x2."""
        x1 = self.up(x1)
        # Input tensor is [Batch, Channel, Height, Width]
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 convolution projecting intermediate features to class logits."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Configurable U-Net for Sentinel-1 SAR Landslide Baseline Segmentation.

    Args:
        in_channels: Number of input channels (e.g. 2 for Sentinel-1 SAR VV+VH,
                     or 4 for pre/post multi-temporal composite). Default is 2.
        out_channels: Number of target classes/logits (default 1 for binary landslide mask).
        features: Channel widths for encoder stages (default [32, 64, 128, 256]).
        bilinear: If True, uses bilinear interpolation for upsampling (lightweight);
                  if False, uses transposed convolutions.
        dropout: Spatial dropout rate between convolution layers.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        features: Optional[List[int]] = None,
        bilinear: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        if len(features) < 2:
            raise ValueError(f"features must have at least 2 stages, got {features}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = list(features)
        self.bilinear = bilinear
        self.dropout = dropout

        # Initial stem
        self.inc = DoubleConv(in_channels, features[0], dropout=dropout)

        # Downscaling stages
        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(
                Down(features[i], features[i + 1], dropout=dropout)
            )

        # Upscaling stages
        self.ups = nn.ModuleList()
        rev_features = list(reversed(features))
        for i in range(len(rev_features) - 1):
            curr_in = rev_features[i] if i == 0 else rev_features[i]
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

        # Final projection to logits
        self.outc = OutConv(features[0], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Logits tensor of shape [B, out_channels, H, W] (raw logits, no sigmoid applied)
        """
        # Encoder
        x_skip = [self.inc(x)]
        for down in self.downs:
            x_skip.append(down(x_skip[-1]))

        # Decoder with skip connections
        x_out = x_skip[-1]
        for i, up in enumerate(self.ups):
            skip = x_skip[-(i + 2)]
            x_out = up(x_out, skip)

        logits = self.outc(x_out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return probabilities in [0, 1]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        """Return total count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
