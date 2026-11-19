"""Temporal SAR ConvLSTM Segmentation Architecture for GeoTrustNet Phase 3.

Integrates a time-distributed 2D CNN spatial encoder, a sequence-masked ConvLSTM
temporal aggregator at the bottleneck, and a multiscale spatial decoder with skip connections.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geotrustnet.models.unet import DoubleConv, Down, Up, OutConv
from src.geotrustnet.models.convlstm import ConvLSTM


class TemporalSARUNet(nn.Module):
    """Spatio-Temporal ConvLSTM U-Net for Sentinel-1 SAR Time-Series Landslide Segmentation.

    Processes 5D SAR sequences [B, T, 2, H, W] through:
      1. Time-distributed multi-scale spatial encoder
      2. Bottleneck ConvLSTM sequence aggregator with missing-observation masking
      3. Multi-scale spatial decoder with skip connections
      4. Binary segmentation logits output [B, 1, H, W]

    Args:
        in_channels: SAR input channels per time step (default 2: VV, VH).
        out_channels: Output segmentation classes/logits (default 1).
        features: Channel widths for encoder stages (default [32, 64, 128, 256]).
        convlstm_layers: Number of stacked ConvLSTM layers at bottleneck (default 1).
        bilinear: Whether to use bilinear upsampling in decoder (default True).
        dropout: Dropout rate across convolutional and temporal layers (default 0.1).
        aggregation: Temporal feature aggregation method: 'last', 'mean', or 'attention' (default 'last').
        skip_aggregation: How to aggregate skip connections across time: 'last' or 'mean' (default 'last').
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        features: Optional[List[int]] = None,
        convlstm_layers: int = 1,
        bilinear: bool = True,
        dropout: float = 0.1,
        aggregation: str = "last",
        skip_aggregation: str = "last",
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
        self.aggregation = aggregation
        self.skip_aggregation = skip_aggregation

        # 1. Multi-scale Spatial Encoder (applied per time step)
        self.inc = DoubleConv(in_channels, features[0], dropout=dropout)
        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(
                Down(features[i], features[i + 1], dropout=dropout)
            )

        # 2. ConvLSTM Temporal Aggregator at the Bottleneck
        bottleneck_channels = features[-1]
        self.convlstm = ConvLSTM(
            in_channels=bottleneck_channels,
            hidden_channels=bottleneck_channels,
            kernel_size=3,
            num_layers=convlstm_layers,
            batch_first=True,
            bias=True,
        )

        # Optional attention pooling across time
        if aggregation == "attention":
            self.temporal_attn = nn.Conv2d(bottleneck_channels, 1, kernel_size=1)

        # 3. Multi-scale Spatial Decoder with Skip Connections
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

        # 4. Final 1x1 projection to class logits
        self.outc = OutConv(features[0], out_channels)

    def _encode_time_distributed(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run 2D CNN encoder over all B*T time steps efficiently.

        Args:
            x: [B, T, C, H, W]

        Returns:
            bottleneck: [B, T, features[-1], H_bot, W_bot]
            skips: List of multi-scale skip tensors, each shaped [B, T, F_i, H_i, W_i]
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        # Stem
        s0 = self.inc(x_flat)
        skips_flat = [s0]

        cur = s0
        for down in self.downs:
            cur = down(cur)
            skips_flat.append(cur)

        bottleneck_flat = skips_flat[-1]
        other_skips_flat = skips_flat[:-1]

        # Reshape bottleneck back to [B, T, F_bot, H_bot, W_bot]
        _, F_bot, H_bot, W_bot = bottleneck_flat.shape
        bottleneck_seq = bottleneck_flat.view(B, T, F_bot, H_bot, W_bot)

        # Reshape skips back to [B, T, F_i, H_i, W_i]
        skips_seq = []
        for s in other_skips_flat:
            _, F_s, H_s, W_s = s.shape
            skips_seq.append(s.view(B, T, F_s, H_s, W_s))

        return bottleneck_seq, skips_seq

    def _aggregate_temporal(
        self,
        lstm_out: torch.Tensor,
        last_states: List[Tuple[torch.Tensor, torch.Tensor]],
        valid_masks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate sequence of ConvLSTM states into single spatial bottleneck map [B, F, H_b, W_b]."""
        B, T, C, H_b, W_b = lstm_out.shape

        if self.aggregation == "last":
            # Use last layer's hidden state (already handles masked freeze if provided)
            return last_states[-1][0]

        elif self.aggregation == "mean":
            if valid_masks is not None:
                # Masked mean over valid steps
                if valid_masks.ndim == 2:
                    m = valid_masks.view(B, T, 1, 1, 1).float()
                elif valid_masks.ndim == 5:
                    m = valid_masks.float()
                    if m.shape[-2:] != (H_b, W_b):
                        m_flat = m.view(B * T, m.shape[2], m.shape[3], m.shape[4])
                        m_flat = F.interpolate(m_flat, size=(H_b, W_b), mode="nearest")
                        m = m_flat.view(B, T, m.shape[2], H_b, W_b)
                else:
                    m = torch.ones_like(lstm_out)
                sum_h = torch.sum(lstm_out * m, dim=1)
                denom = torch.clamp(torch.sum(m, dim=1), min=1.0)
                return sum_h / denom
            else:
                return torch.mean(lstm_out, dim=1)

        elif self.aggregation == "attention":
            # Spatial-averaged temporal score
            weights = []
            for t in range(T):
                h_t = lstm_out[:, t, :, :, :]
                # Conv2d to 1 channel then spatial mean -> [B, 1]
                score = self.temporal_attn(h_t).mean(dim=[-2, -1])  # [B, 1]
                weights.append(score)
            weights = torch.stack(weights, dim=1)  # [B, T, 1]

            if valid_masks is not None:
                if valid_masks.ndim == 2:
                    mask_2d = valid_masks.view(B, T, 1)
                else:
                    mask_2d = valid_masks.view(B, T, -1).mean(dim=-1, keepdim=True) > 0
                weights = weights.masked_fill(~mask_2d, -1e9)

            attn_probs = F.softmax(weights, dim=1).view(B, T, 1, 1, 1)
            aggregated = torch.sum(lstm_out * attn_probs, dim=1)
            return aggregated

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def _aggregate_skips(
        self,
        skips_seq: List[torch.Tensor],
        valid_masks: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Aggregate skip connections across time steps to feed decoder."""
        aggregated_skips = []
        for skip in skips_seq:
            # skip shape is [B, T, F_s, H_s, W_s]
            B, T, F_s, H_s, W_s = skip.shape
            if self.skip_aggregation == "last":
                # Take the last time slice (e.g. most recent valid observation)
                agg = skip[:, -1, :, :, :]
            elif self.skip_aggregation == "mean":
                if valid_masks is not None:
                    if valid_masks.ndim == 2:
                        m = valid_masks.view(B, T, 1, 1, 1).float()
                    elif valid_masks.ndim == 5:
                        m = valid_masks.float()
                        if m.shape[-2:] != (H_s, W_s):
                            m_flat = m.view(B * T, m.shape[2], m.shape[3], m.shape[4])
                            m_flat = F.interpolate(m_flat, size=(H_s, W_s), mode="nearest")
                            m = m_flat.view(B, T, m.shape[2], H_s, W_s)
                    else:
                        m = torch.ones((B, T, 1, 1, 1), device=skip.device)
                    agg = torch.sum(skip * m, dim=1) / torch.clamp(torch.sum(m, dim=1), min=1.0)
                else:
                    agg = torch.mean(skip, dim=1)
            else:
                agg = skip[:, -1, :, :, :]
            aggregated_skips.append(agg)
        return aggregated_skips

    def forward(
        self,
        x: torch.Tensor,
        valid_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass over temporal SAR sequence.

        Args:
            x: Input tensor of shape [B, T, C, H, W]
               If 4D [B, C, H, W] is passed, unsqueezes to [B, 1, C, H, W].
            valid_masks: Optional validity mask [B, T] or [B, T, 1, H, W].

        Returns:
            Logits tensor of shape [B, out_channels, H, W]
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)  # Add T=1 dimension

        B, T, C, H, W = x.shape

        # 1. Time-distributed spatial encoding
        bottleneck_seq, skips_seq = self._encode_time_distributed(x)

        # 2. ConvLSTM temporal aggregation at bottleneck
        lstm_out, last_states = self.convlstm(bottleneck_seq, valid_masks=valid_masks)
        bottleneck_agg = self._aggregate_temporal(lstm_out, last_states, valid_masks=valid_masks)

        # 3. Aggregate skip connections
        agg_skips = self._aggregate_skips(skips_seq, valid_masks=valid_masks)

        # 4. Decoder with skip connections (innermost to outermost)
        cur = bottleneck_agg
        for i, up in enumerate(self.ups):
            skip = agg_skips[-(i + 1)]
            cur = up(cur, skip)

        logits = self.outc(cur)
        return logits

    def predict_proba(
        self,
        x: torch.Tensor,
        valid_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run forward pass and return predicted probabilities in [0, 1]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, valid_masks=valid_masks)
            return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        """Return total count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
