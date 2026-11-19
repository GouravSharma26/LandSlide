"""GeoTrustNet Core Model: Reliability-Aware Spatio-Temporal Multimodal Deep Learning with Uncertainty.

Integrates independent sensor encoders, adaptive reliability-aware fusion,
and Monte Carlo Dropout for epistemic uncertainty quantification.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geotrustnet.models.unet import Up, OutConv
from src.geotrustnet.models.encoders import SAREncoder, OpticalEncoder, TerrainEncoder
from src.geotrustnet.models.reliability import ReliabilityAwareFusion
from src.geotrustnet.models.convlstm import ConvLSTM


class GeoTrustNet(nn.Module):
    """GeoTrustNet: Uncertainty-Aware Multimodal Landslide Segmentation Network.

    Args:
        features: Channel progression across encoder/decoder stages (default [32, 64, 128, 256]).
        sar_channels: Channels for Sentinel-1 SAR (default 2: VV, VH).
        optical_channels: Channels for Sentinel-2 Optical (default 10: spectral bands).
        dem_channels: Channels for Copernicus DEM (default 2: elevation, slope).
        out_channels: Output logits channels (default 1 for binary landslide mask).
        bilinear: Whether to use bilinear upsampling in decoder (default True).
        dropout: Dropout rate for regularization and Monte Carlo epistemic uncertainty (default 0.2).
        spatial_reliability: Whether reliability weights vary spatially (default True).
    """

    def __init__(
        self,
        features: Optional[List[int]] = None,
        sar_channels: int = 2,
        optical_channels: int = 10,
        dem_channels: int = 2,
        out_channels: int = 1,
        bilinear: bool = True,
        dropout: float = 0.2,
        spatial_reliability: bool = True,
        use_temporal: bool = False,
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
        self.spatial_reliability = spatial_reliability
        self.use_temporal = use_temporal

        # 1. Independent Sensor Encoders
        self.sar_encoder = SAREncoder(in_channels=sar_channels, features=features, dropout=dropout)
        self.optical_encoder = OpticalEncoder(in_channels=optical_channels, features=features, dropout=dropout)
        self.dem_encoder = TerrainEncoder(in_channels=dem_channels, features=features, dropout=dropout)

        if self.use_temporal:
            self.sar_convlstm = ConvLSTM(
                in_channels=features[-1],
                hidden_channels=features[-1],
                kernel_size=3,
                num_layers=1,
                batch_first=True,
                bias=True,
            )

        # 2. Multi-Scale Reliability-Aware Fusion Modules
        self.skip_fusions = nn.ModuleList()
        for f_dim in features[:-1]:
            self.skip_fusions.append(
                ReliabilityAwareFusion(
                    feature_dim=f_dim,
                    num_modalities=3,
                    out_dim=f_dim,
                    dropout=dropout,
                    spatial=spatial_reliability,
                )
            )

        self.bottleneck_fusion = ReliabilityAwareFusion(
            feature_dim=features[-1],
            num_modalities=3,
            out_dim=features[-1],
            dropout=dropout,
            spatial=spatial_reliability,
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

        # 4. Final Projection to Binary Logits
        self.outc = OutConv(features[0], out_channels)

    def _prepare_modality(
        self,
        tensor: Optional[torch.Tensor],
        expected_channels: int,
        ref_shape: Tuple[int, int, int],
        device: torch.device,
        has_flag: Optional[torch.Tensor] = None,
        quality_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Validate, reshape, or instantiate fallback for an input sensor."""
        B, H, W = ref_shape

        if tensor is None:
            x = torch.zeros((B, expected_channels, H, W), device=device, dtype=torch.float32)
            avail = torch.zeros(B, device=device, dtype=torch.bool)
            q = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
            return x, avail, q

        if tensor.ndim == 5:
            tensor = tensor[:, -1, :, :, :]

        x = tensor.to(device).float()

        if has_flag is not None:
            avail = has_flag.to(device).bool()
            if avail.ndim > 1:
                avail = avail.squeeze()
        else:
            avail = torch.ones(B, device=device, dtype=torch.bool)

        # Quality map
        if quality_tensor is not None:
            if quality_tensor.ndim == 5:
                quality_tensor = quality_tensor[:, -1, :, :, :]
            q = quality_tensor.to(device).float()
        else:
            q = torch.ones((B, 1, H, W), device=device, dtype=torch.float32)

        return x, avail, q

    def forward(
        self,
        sar: Optional[torch.Tensor] = None,
        optical: Optional[torch.Tensor] = None,
        dem: Optional[torch.Tensor] = None,
        has_sar: Optional[torch.Tensor] = None,
        has_optical: Optional[torch.Tensor] = None,
        has_dem: Optional[torch.Tensor] = None,
        sar_quality: Optional[torch.Tensor] = None,
        optical_quality: Optional[torch.Tensor] = None,
        dem_quality: Optional[torch.Tensor] = None,
        return_reliability: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass with reliability-aware feature fusion.

        Args:
            sar: Sentinel-1 SAR [B, 2, H, W]
            optical: Sentinel-2 Optical [B, 10, H, W]
            dem: Copernicus DEM [B, 2, H, W]
            has_sar: Availability flag [B] for SAR
            has_optical: Availability flag [B] for Optical
            has_dem: Availability flag [B] for DEM
            sar_quality: Optional quality/valid mask for SAR
            optical_quality: Optional cloud/valid mask for Optical
            dem_quality: Optional validity mask for DEM
            return_reliability: If True, returns (logits, dict of reliability weights)

        Returns:
            logits: [B, 1, H, W] (raw unnormalized logits)
            reliability_dict: Dictionary containing normalized weights per modality
        """
        ref = sar if sar is not None else (optical if optical is not None else dem)
        if ref is None:
            raise ValueError("At least one modality (sar, optical, or dem) must be provided.")

        B = ref.size(0)
        H, W = ref.size(-2), ref.size(-1)
        device = ref.device
        ref_shape = (B, H, W)

        # Prepare inputs and quality signals
        if self.use_temporal and sar is not None and sar.ndim == 5:
            # Spatio-temporal ConvLSTM sequence feature extraction for 5D SAR [B, T, C, H, W]
            B_s, T_s, C_s, H_s, W_s = sar.shape
            sar_flat = sar.reshape(B_s * T_s, C_s, H_s, W_s).float()
            bot_flat, skips_flat = self.sar_encoder(sar_flat)
            F_bot, H_b, W_b = bot_flat.shape[1], bot_flat.shape[2], bot_flat.shape[3]
            bot_seq = bot_flat.reshape(B_s, T_s, F_bot, H_b, W_b)
            _, last_states = self.sar_convlstm(bot_seq)
            bot_sar = last_states[-1][0]
            # Aggregate skips using latest time step
            skips_sar = [
                s.reshape(B_s, T_s, s.shape[1], s.shape[2], s.shape[3])[:, -1]
                for s in skips_flat
            ]
            if has_sar is not None:
                a_sar = has_sar.to(device).bool()
                if a_sar.ndim > 1:
                    a_sar = a_sar.squeeze()
            else:
                a_sar = torch.ones(B, device=device, dtype=torch.bool)
            if sar_quality is not None:
                q_sar = (sar_quality[:, -1] if sar_quality.ndim == 5 else sar_quality).to(device).float()
            else:
                q_sar = torch.ones((B, 1, H, W), device=device, dtype=torch.float32)
        else:
            x_sar, a_sar, q_sar = self._prepare_modality(sar, self.sar_channels, ref_shape, device, has_sar, sar_quality)
            bot_sar, skips_sar = self.sar_encoder(x_sar, is_available=a_sar)

        x_opt, a_opt, q_opt = self._prepare_modality(optical, self.optical_channels, ref_shape, device, has_optical, optical_quality)
        x_dem, a_dem, q_dem = self._prepare_modality(dem, self.dem_channels, ref_shape, device, has_dem, dem_quality)

        avails = [a_sar, a_opt, a_dem]
        qualities = [q_sar, q_opt, q_dem]

        # 1. Feature Extraction per Modality
        bot_opt, skips_opt = self.optical_encoder(x_opt, is_available=a_opt)
        bot_dem, skips_dem = self.dem_encoder(x_dem, is_available=a_dem)

        # 2. Multi-Scale Reliability-Aware Fusion
        fused_skips = []
        skip_weights_records = []
        for i, skip_fusion_mod in enumerate(self.skip_fusions):
            scale_feats = [skips_sar[i], skips_opt[i], skips_dem[i]]
            fused_s, w_s = skip_fusion_mod(scale_feats, avails, qualities)
            fused_skips.append(fused_s)
            skip_weights_records.append(w_s)

        # Bottleneck Fusion
        bot_feats = [bot_sar, bot_opt, bot_dem]
        fused_bottleneck, bot_weights = self.bottleneck_fusion(bot_feats, avails, qualities)

        # 3. Spatial Decoder
        cur = fused_bottleneck
        for i, up in enumerate(self.ups):
            skip = fused_skips[-(i + 1)]
            cur = up(cur, skip)

        logits = self.outc(cur)

        if return_reliability:
            reliability_dict = {
                "weight_sar": bot_weights[0],
                "weight_optical": bot_weights[1],
                "weight_dem": bot_weights[2],
            }
            return logits, reliability_dict

        return logits

    def enable_mc_dropout(self) -> None:
        """Activate Dropout layers while keeping BatchNorm layers in evaluation mode."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    @torch.no_grad()
    def predict_uncertainty(
        self,
        sar: Optional[torch.Tensor] = None,
        optical: Optional[torch.Tensor] = None,
        dem: Optional[torch.Tensor] = None,
        has_sar: Optional[torch.Tensor] = None,
        has_optical: Optional[torch.Tensor] = None,
        has_dem: Optional[torch.Tensor] = None,
        sar_quality: Optional[torch.Tensor] = None,
        optical_quality: Optional[torch.Tensor] = None,
        dem_quality: Optional[torch.Tensor] = None,
        num_mc_samples: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """Estimate predictive mean, epistemic variance, and entropy via Monte Carlo Dropout.

        Args:
            num_mc_samples: Number of stochastic forward passes (default 10).

        Returns:
            Dictionary containing:
              - 'mean_prob': [B, 1, H, W]
              - 'variance': [B, 1, H, W] (epistemic uncertainty map)
              - 'entropy': [B, 1, H, W] (predictive information entropy)
              - 'confidence': [B, 1, H, W] (|2*p - 1|)
              - 'reliability_sar': [B, 1, H_b, W_b]
              - 'reliability_optical': [B, 1, H_b, W_b]
              - 'reliability_dem': [B, 1, H_b, W_b]
        """
        self.eval()
        self.enable_mc_dropout()

        probs_list = []
        rel_sar_list = []
        rel_opt_list = []
        rel_dem_list = []

        for _ in range(num_mc_samples):
            logits, rel_dict = self.forward(
                sar=sar,
                optical=optical,
                dem=dem,
                has_sar=has_sar,
                has_optical=has_optical,
                has_dem=has_dem,
                sar_quality=sar_quality,
                optical_quality=optical_quality,
                dem_quality=dem_quality,
                return_reliability=True,
            )
            probs = torch.sigmoid(logits)
            probs_list.append(probs)
            rel_sar_list.append(rel_dict["weight_sar"])
            rel_opt_list.append(rel_dict["weight_optical"])
            rel_dem_list.append(rel_dict["weight_dem"])

        # Stack across MC passes: [N, B, 1, H, W]
        stacked_probs = torch.stack(probs_list, dim=0)

        # 1. Predictive Mean
        mean_prob = torch.mean(stacked_probs, dim=0)

        # 2. Epistemic Variance (Uncertainty)
        variance = torch.var(stacked_probs, dim=0, unbiased=False)

        # 3. Predictive Entropy
        eps = 1e-7
        p_clamped = torch.clamp(mean_prob, eps, 1.0 - eps)
        entropy = -(p_clamped * torch.log2(p_clamped) + (1.0 - p_clamped) * torch.log2(1.0 - p_clamped))

        # 4. Confidence Map
        confidence = torch.abs(2.0 * mean_prob - 1.0)

        # Mean reliability weights
        mean_rel_sar = torch.stack(rel_sar_list, dim=0).mean(dim=0)
        mean_rel_opt = torch.stack(rel_opt_list, dim=0).mean(dim=0)
        mean_rel_dem = torch.stack(rel_dem_list, dim=0).mean(dim=0)

        return {
            "mean_prob": mean_prob,
            "variance": variance,
            "entropy": entropy,
            "confidence": confidence,
            "reliability_sar": mean_rel_sar,
            "reliability_optical": mean_rel_opt,
            "reliability_dem": mean_rel_dem,
        }

    def count_parameters(self) -> int:
        """Return total count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
