"""Reliability-aware modality fusion modules for GeoTrustNet.

Implements adaptive modality reliability estimation, masked softmax weight normalization,
and availability-conditioned feature fusion under partial observation and sensor degradation.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geotrustnet.models.unet import DoubleConv


class ModalityReliabilityEstimator(nn.Module):
    """Estimates spatial or patch-level reliability scores for an individual sensor modality.

    Args:
        feature_dim: Number of feature channels in the input representation.
        hidden_dim: Number of intermediate channels in the scoring network.
        spatial: If True, computes a pixel-wise 2D reliability map [B, 1, H, W].
                 If False, computes a single patch-level scalar [B, 1, 1, 1].
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
        spatial: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial = spatial
        hidden = hidden_dim if hidden_dim is not None else max(16, feature_dim // 4)

        # Input is feature channels + 1 quality channel
        self.scorer = nn.Sequential(
            nn.Conv2d(feature_dim + 1, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        quality: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Estimate raw reliability scores.

        Args:
            x: Modality feature tensor [B, C, H, W]
            quality: Optional quality signal [B] or [B, 1, H, W] (e.g., clear-sky ratio, validity).
                     Defaults to 1.0 (fully valid) if None.

        Returns:
            Raw unnormalized reliability logits [B, 1, H, W] or [B, 1, 1, 1]
        """
        batch_size, _, height, width = x.shape
        device = x.device

        # Standardize quality signal to [B, 1, H, W]
        if quality is None:
            q_map = torch.ones((batch_size, 1, height, width), device=device, dtype=x.dtype)
        elif quality.ndim == 1:
            q_map = quality.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width).float()
        elif quality.ndim == 2:
            q_map = quality.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width).float()
        else:
            q_map = quality.float()
            if q_map.shape[-2:] != (height, width):
                q_map = F.interpolate(q_map, size=(height, width), mode="nearest")

        combined = torch.cat([x, q_map], dim=1)
        scores = self.scorer(combined)  # [B, 1, H, W]

        if not self.spatial:
            scores = scores.mean(dim=[-2, -1], keepdim=True)  # [B, 1, 1, 1]

        return scores


class ReliabilityAwareFusion(nn.Module):
    """Adaptive reliability-aware multimodal fusion module.

    Dynamically reweights modalities according to estimated reliability scores and sensor quality,
    enforcing strictly zero weight on missing modalities via masked softmax.

    Args:
        feature_dim: Channel dimension of input representations.
        num_modalities: Number of candidate modalities (default 3: SAR, Optical, DEM).
        out_dim: Output channel dimension (default matches feature_dim).
        dropout: Spatial dropout rate in projection and refinement layers.
        spatial: Whether to compute spatially varying reliability maps (default True).
    """

    def __init__(
        self,
        feature_dim: int,
        num_modalities: int = 3,
        out_dim: Optional[int] = None,
        dropout: float = 0.1,
        spatial: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.out_dim = out_dim if out_dim is not None else feature_dim
        self.spatial = spatial

        # Modality reliability estimators
        self.estimators = nn.ModuleList([
            ModalityReliabilityEstimator(feature_dim, spatial=spatial)
            for _ in range(num_modalities)
        ])

        # Per-modality feature projection layers
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, self.out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_modalities)
        ])

        # Availability indicator conditioning + Refinement DoubleConv
        # Input: out_dim (weighted sum) + num_modalities (indicator maps)
        refinement_in = self.out_dim + num_modalities
        self.refinement = DoubleConv(
            in_channels=refinement_in,
            out_channels=self.out_dim,
            dropout=dropout,
        )

    def forward(
        self,
        features: List[torch.Tensor],
        availabilities: List[torch.Tensor],
        qualities: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Compute reliability-weighted feature fusion.

        Args:
            features: List of tensors [z_1, ..., z_M], each [B, C, H, W]
            availabilities: List of boolean/binary flags [a_1, ..., a_M], each [B]
            qualities: Optional list of quality scores [q_1, ..., q_M]

        Returns:
            fused_features: [B, out_dim, H, W]
            normalized_weights: List of normalized reliability weight maps [w_1, ..., w_M],
                                each [B, 1, H, W] or [B, 1, 1, 1]
        """
        if len(features) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} features, got {len(features)}")

        batch_size, _, height, width = features[0].shape
        device = features[0].device

        if qualities is None:
            qualities = [None] * self.num_modalities

        # 1. Estimate raw reliability scores per modality
        raw_scores = []
        for i in range(self.num_modalities):
            s_i = self.estimators[i](features[i], qualities[i])
            raw_scores.append(s_i)  # [B, 1, H, W]

        # Stack scores: [B, M, H, W]
        scores_stacked = torch.cat(raw_scores, dim=1)

        # 2. Prepare availability masks: [B, M, 1, 1]
        avails_list = []
        for a in availabilities:
            if a.ndim == 1:
                avails_list.append(a.view(batch_size, 1, 1, 1).float())
            else:
                avails_list.append(a.view(batch_size, 1, 1, 1).float())
        avails_stacked = torch.cat(avails_list, dim=1).to(device)  # [B, M, 1, 1]

        # 3. Masked Softmax over modalities:
        # Where a modality is unavailable (avails == 0), set logit to -1e4
        masked_logits = scores_stacked.masked_fill(avails_stacked == 0, -10000.0)

        # Handle samples where all modalities might be missing
        num_available = avails_stacked.sum(dim=1, keepdim=True)  # [B, 1, 1, 1]
        any_available = (num_available > 0).float()

        weights_stacked = F.softmax(masked_logits, dim=1) * any_available  # [B, M, H, W]

        # Extract normalized weights per modality
        normalized_weights = [
            weights_stacked[:, i : i + 1, :, :] for i in range(self.num_modalities)
        ]

        # 4. Feature projection and weighted combination
        weighted_features_sum = torch.zeros(
            (batch_size, self.out_dim, height, width),
            device=device,
            dtype=features[0].dtype,
        )

        for i in range(self.num_modalities):
            proj_i = self.projections[i](features[i])
            w_i = normalized_weights[i]
            weighted_features_sum = weighted_features_sum + (proj_i * w_i)

        # 5. Availability indicator conditioning
        avails_spatial = avails_stacked.expand(batch_size, self.num_modalities, height, width)
        combined_refine = torch.cat([weighted_features_sum, avails_spatial], dim=1)
        fused = self.refinement(combined_refine)

        return fused, normalized_weights
