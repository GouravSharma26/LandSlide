# GeoTrustNet Research Design Document — Phase 5A Audit & Formulation

**Project:** GeoTrustNet: Uncertainty-Aware Spatio-Temporal Multimodal Deep Learning for Reliable Landslide Detection and Geographic Generalization  
**Phase:** 5A — Architecture and Research Design Audit  
**Date:** November 2026  
**Status:** Approved & Formulated  

---

## 1. Research Motivation & Core Hypothesis

### The Problem in Conventional EO Fusion
Existing remote-sensing landslide detection architectures treat multimodal data sources (SAR, optical, elevation) using either:
1. Early channel stacking (channel concatenation at raw input resolution), or
2. Fixed feature concatenation (such as Phase 4's `FeatureLevelFusion`).

Both approaches make a flawed assumption: **that all sensors are equally trustworthy across all geographic areas and weather conditions**. In practice:
- **Optical (Sentinel-2)** provides high spectral resolution for vegetation loss and scarp exposure, but suffers complete blackout during monsoonal cloud cover, dense cloud shadows, or heavy aerosol scattering.
- **SAR (Sentinel-1)** penetrates clouds and rain, but suffers from geometric layover, radar foreshortening, and shadow on steep slopes oriented away from the sensor's line of sight.
- **DEM (Copernicus GLO-30)** provides stable gravitational drivers (slope, aspect, elevation), but has static resolution ($30\text{m}$) and lacks dynamic temporal information regarding triggering events.

### The GeoTrustNet Hypothesis
> **Core Hypothesis:** Landslide segmentation robustness, calibration, and geographic generalization will significantly improve if the network dynamically estimates **modality reliability** from observation quality signals (cloud masks, sensor geometry, availability flags) to adaptively reweight modality representations prior to spatial decoding, while simultaneously quantifying **epistemic uncertainty** via Monte Carlo Dropout to identify out-of-distribution terrain and ambiguous boundary predictions.

---

## 2. Audit of Existing Repository Implementations (Phases 1–4)

A comprehensive audit of the repository establishes the modular baseline foundation:

1. **Data Contracts & Splits (`src/geotrustnet/data/`):**
   - Formalized multi-modal patches ($128 \times 128$ at $10\text{m}$ resolution) in `contracts.py` with typed metadata and provenance.
   - Leakage-safe, deterministic geographic and event-held-out splits in `splits.py` with SHA-256 integrity verification.
   - Ground truth labels are strictly binary $\{0, 1\}$, with nodata masked.

2. **Single-Modality Baseline (`src/geotrustnet/models/unet.py` — Phase 2):**
   - 2-channel Sentinel-1 SAR U-Net ($1.95\text{M}$ parameters), bilinear upsampling, combined BCE + Soft Dice loss.

3. **Temporal Baseline (`src/geotrustnet/models/temporal_unet.py` — Phase 3):**
   - 5D spatio-temporal ConvLSTM aggregator ($6.67\text{M}$ parameters) with state-freeze sequence masking ($M_t=0 \implies C_t = C_{t-1}, H_t = H_{t-1}$) to prevent zero-padding noise from corrupting temporal memory.

4. **Conventional Multimodal Baseline (`src/geotrustnet/models/multimodal_unet.py` — Phase 4):**
   - Independent encoders (`SAREncoder`, `OpticalEncoder`, `TerrainEncoder`) + `FeatureLevelFusion` ($7.44\text{M}$ parameters) using channel concatenation and 3 availability indicator channels.

---

## 3. Physical Reliability Signals in Real Data

The table below catalogs which quality signals are genuinely available from satellite data products versus derived mathematically:

| Modality | Quality Signal Source | Signal Representation | Mathematical Interpretation |
|---|---|---|---|
| **Optical** | Sentinel-2 Scene Classification (SCL) / Cloud Mask | Binary mask `optical_valid` $\in \{0, 1\}^{B \times 1 \times H \times W}$ | $0 = \text{Cloud/Shadow/Defective}, 1 = \text{Clear Surface}$ |
| **Optical** | Patch-Level Valid Coverage | $q_{\text{opt}} = \frac{1}{HW}\sum_{x,y} M_{\text{opt}}(x, y) \in [0, 1]$ | Clear-sky spatial ratio (0.0 = total overcast, 1.0 = clear) |
| **Optical** | Acquisition Availability | $A_{\text{opt}} \in \{0, 1\}$ | Sensor revisit availability (0 = missing acquisition) |
| **SAR** | Radar Layover / Shadow / Nodata Mask | Binary mask `sar_valid` $\in \{0, 1\}^{B \times 1 \times H \times W}$ | $0 = \text{Geometric distortion/nodata}, 1 = \text{Valid backscatter}$ |
| **SAR** | Patch-Level Valid Coverage | $q_{\text{sar}} = \frac{1}{HW}\sum_{x,y} M_{\text{sar}}(x, y) \in [0, 1]$ | Valid measurement ratio |
| **SAR** | Acquisition Availability | $A_{\text{sar}} \in \{0, 1\}$ | Sensor revisit availability |
| **Terrain** | Physical Slope & Elevation Bounds | Physical plausibility mask $M_{\text{dem}}$ | Elevation $\ge 0\text{m}$, Slope $\in [0^\circ, 90^\circ]$ |
| **Terrain** | Acquisition Availability | $A_{\text{dem}} \in \{0, 1\}$ | DEM coverage flag |

> [!IMPORTANT]
> Quality signals are strictly derived from real observation properties or sensor masks. No quality flags are fabricated or assumed.

---

## 4. Mathematical Formulation: Reliability-Aware Modality Fusion

### 4.1 Modality Reliability Scoring
For each candidate modality $m \in \{\text{sar}, \text{optical}, \text{dem}\}$, let $Z_m \in \mathbb{R}^{B \times C \times H_b \times W_b}$ denote the encoded feature representation at the bottleneck, $q_m \in [0, 1]$ denote its valid spatial ratio, and $A_m \in \{0, 1\}$ denote its availability flag.

The raw spatial reliability map $r_m(x, y)$ is estimated by a lightweight convolutional scoring network $\psi_m$:
$$r_m(x, y) = \psi_m\left([Z_m \,\|\, \mathbf{Q}_m]\right) \in \mathbb{R}^{B \times 1 \times H_b \times W_b}$$
where $\mathbf{Q}_m$ is the quality indicator map expanded to the spatial dimension.

### 4.2 Masked Softmax Normalization
To guarantee that missing modalities receive **strictly zero weight** ($w_m = 0.0$) and that weights across all available modalities $\mathcal{A}(x) = \{m \mid A_m = 1\}$ sum to $1.0$:

$$\tilde{r}_m(x, y) = \begin{cases} r_m(x, y) & \text{if } A_m = 1 \\ -\infty & \text{if } A_m = 0 \end{cases}$$

$$w_m(x, y) = \frac{\exp(\tilde{r}_m(x, y))}{\sum_{j=1}^M \exp(\tilde{r}_j(x, y)) + \epsilon}$$

#### Edge Cases Handled Mathematically:
1. **Single Modality Available ($|\mathcal{A}| = 1$):**  
   If only SAR is present ($A_{\text{sar}}=1, A_{\text{opt}}=0, A_{\text{dem}}=0$), then $\tilde{r}_{\text{opt}} = -\infty$ and $\tilde{r}_{\text{dem}} = -\infty$. Consequently, $w_{\text{sar}}(x, y) \equiv 1.0$ everywhere.
2. **All Modalities Unavailable ($|\mathcal{A}| = 0$):**  
   All $w_m(x, y) \equiv 0.0$, preventing any division-by-zero or NaN propagation.
3. **Cloudy Optical vs Clean SAR:**  
   When optical has high cloud contamination ($q_{\text{opt}} \to 0$), the network learns $r_{\text{opt}} \ll r_{\text{sar}}$, automatically dynamically transferring fusion authority to SAR and DEM.

### 4.3 Feature Fusion
The fused representation combines reliability-weighted features and availability indicators:
$$F_{\text{fused}} = \sum_{m=1}^M w_m(x, y) \odot \text{Proj}_m(Z_m) + \text{IndicatorMap}(\mathbf{A})$$
Followed by a residual refinement block.

---

## 5. Predictive Uncertainty Methodology

### 5.1 Epistemic Uncertainty via Monte Carlo Dropout
To quantify model uncertainty arising from ambiguous scarp boundaries or out-of-distribution terrain, spatial dropout layers ($p = 0.2$) are retained active at inference time.

For an input sample $x$, $N=10$ stochastic forward passes produce predictive probabilities:
$$p^{(i)} = \sigma(f_{\theta^{(i)}}(x)), \quad i = 1, \dots, N$$

We compute:
1. **Mean Predictive Probability:**
   $$\mu(x, y) = \frac{1}{N} \sum_{i=1}^N p^{(i)}(x, y)$$
2. **Epistemic Predictive Variance (Uncertainty Map):**
   $$\sigma^2(x, y) = \frac{1}{N} \sum_{i=1}^N \left(p^{(i)}(x, y) - \mu(x, y)\right)^2$$
3. **Predictive Entropy:**
   $$\mathcal{H}(x, y) = -\mu \log_2(\mu + \epsilon) - (1 - \mu) \log_2(1 - \mu + \epsilon)$$

### 5.2 Selective Prediction & Calibration Metrics
- **Expected Calibration Error (ECE):** Partitions predicted probabilities into $K=10$ bins $B_k \subset [0, 1]$ and computes:
  $$\text{ECE} = \sum_{k=1}^K \frac{|B_k|}{N_{\text{pixels}}} \left| \text{acc}(B_k) - \text{conf}(B_k) \right|$$
- **Brier Score:** Mean squared error between probabilities and ground truth binary labels.
- **Selective Prediction Analysis:** Measures segmentation accuracy (IoU, Dice) as high-uncertainty pixels ($\sigma^2 > \tau$) are rejected, quantifying the reliability of the uncertainty signal.
