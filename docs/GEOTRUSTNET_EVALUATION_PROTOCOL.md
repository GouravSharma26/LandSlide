# GeoTrustNet Experimental & Evaluation Protocol

**Research Title:** GeoTrustNet: Uncertainty-Aware Spatio-Temporal Multimodal Deep Learning for Reliable Landslide Detection and Geographic Generalization  
**Document:** Formal Scientific Evaluation and Benchmarking Protocol  
**Version:** 1.0.0  
**Date:** 2026-11-19  

---

## 1. Research Objectives & Hypotheses

### Primary Research Hypothesis ($H_1$)
Conventional multimodal fusion networks that use static feature concatenation or unweighted sensor summation are vulnerable to localized sensor failure (e.g., optical cloud contamination, SAR radar shadows, or missing modalities). An **adaptive spatial reliability weighting mechanism** that dynamically scales modality representations according to their observed quality will achieve:
1. Higher segmentation accuracy (Dice / IoU) under adverse observational conditions.
2. Graceful degradation when optical or SAR modalities are partially or completely absent.
3. Quantifiable epistemic uncertainty that flags ambiguous or unreliable predictions.

### Null Hypothesis ($H_0$)
Dynamic reliability-aware feature reweighting provides no statistically significant improvement in segmentation performance or calibration over conventional multimodal concatenation baselines.

---

## 2. Leakage-Safe Data Partitioning Protocol

### 2.1 Spatial and Event Holdout Rule
To evaluate true geographic generalization and prevent spatial autocorrelation leakage:
1. **Disjoint Geographic Groups:** All patches originating from the same geographic region (`region_id`) must be assigned exclusively to a single split (Train, Validation, or Test).
2. **Disjoint Event Groups:** All patches recording the same triggering meteorological or seismic event (`event_id`) must remain strictly within the same split.
3. **Partition Ratios:**
   - **Training Set (70%):** Model parameter optimization via gradient descent.
   - **Validation Set (15%):** Model checkpoint selection, learning rate scheduling, early stopping, and temperature scaling/calibration fitting.
   - **Test Set (15%):** Strictly held-out benchmark evaluation. **No hyperparameter tuning or architectural decisions may be made based on test set scores.**

### 2.2 Mathematical Definition of Leakage-Free Partitions
Let $\mathcal{R}_{\text{train}}, \mathcal{R}_{\text{val}}, \mathcal{R}_{\text{test}}$ be the sets of region identifiers in each partition. The protocol requires:
$$\mathcal{R}_{\text{train}} \cap \mathcal{R}_{\text{val}} = \emptyset, \quad \mathcal{R}_{\text{train}} \cap \mathcal{R}_{\text{test}} = \emptyset, \quad \mathcal{R}_{\text{val}} \cap \mathcal{R}_{\text{test}} = \emptyset$$

### 2.3 Split Reproducibility
- All split allocations must be generated deterministically using a fixed random seed (`seed = 42`).
- The split manifest file must record an immutable SHA-256 hash verifying partition integrity.

---

## 3. Standardization and Preprocessing

All competing baseline models and GeoTrustNet variants must ingest identically preprocessed inputs:

1. **Sentinel-1 SAR:**
   - Channels: VV, VH backscatter in decibels (dB).
   - Valid clipping range: $[-45.0, 10.0]$ dB.
   - Normalization: Linear affine rescaling to $[-1.0, 1.0]$ via $x_{\text{norm}} = 2 \cdot \frac{x - (-45)}{10 - (-45)} - 1$.
   - Nodata replacement: Sentinel nodata pixels replaced with $-1.0$ and marked with $0$ in the sensor validity mask.

2. **Sentinel-2 Optical:**
   - 10 spectral bands (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12).
   - Valid clipping range: $[0, 10000]$ surface reflectance.
   - Normalization: Scaled to $[0.0, 1.0]$ via $x_{\text{norm}} = \frac{x}{10000.0}$.
   - Cloud masking: Scene Classification Layer (SCL) classes 3 (cloud shadows), 8, 9, 10 (cloud medium/high probability), and 11 (snow) treated as invalid.

3. **Copernicus DEM:**
   - Elevation: Standardized using training set mean $\mu_{\text{elev}}$ and standard deviation $\sigma_{\text{elev}}$.
   - Slope: Valid range $[0.0, 90.0]^\circ$ rescaled to $[0.0, 1.0]$ via $x / 90.0$.

4. **Target Landslide Mask:**
   - Binary encoding: $\{0, 1\}$.
   - Nodata value ($255$) masked out of the loss calculation and evaluation metrics.

---

## 4. Controlled Benchmark Models

The evaluation protocol defines four standardized models:

| Model | Code Designation | Modalities | Input Tensor Shape | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Model A** | `SAR-UNet` | Sentinel-1 SAR | $[B, 2, 128, 128]$ | Single-modality SAR baseline |
| **Model B** | `TemporalSAR-ConvLSTM` | Sentinel-1 SAR Time Series | $[B, T, 2, 128, 128]$ | Spatio-temporal SAR baseline |
| **Model C** | `Multimodal-UNet` | SAR + Optical + DEM | $[B, 2+10+2, 128, 128]$ | Conventional concatenation fusion |
| **Model D** | `GeoTrustNet` | SAR + Optical + DEM | Independent encoders | Proposed reliability-aware fusion |

All models must be trained with:
- Loss: Combined Binary Cross-Entropy (0.5) + Soft Dice (0.5).
- Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-4}$).
- Scheduler: CosineAnnealingLR.
- Early Stopping: Monitored on validation IoU with patience of 10 epochs.

---

## 5. Quantitative Evaluation Metrics

### 5.1 Pixel-Level Segmentation Metrics
Let $TP, FP, TN, FN$ be the counts of true positive, false positive, true negative, and false negative pixels evaluated across valid pixels:
- **Precision:** $\frac{TP}{TP + FP}$
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$
- **Dice Similarity Coefficient (F1):** $\frac{2 \cdot TP}{2 \cdot TP + FP + FN}$
- **Intersection over Union (IoU / Jaccard):** $\frac{TP}{TP + FP + FN}$
- **Precision-Recall AUC (PR-AUC):** Trapezoidal integral of precision-recall curve (primary metric under severe class imbalance).

### 5.2 Probability Calibration Metrics
- **Brier Score:** $\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$
- **Expected Calibration Error (ECE):** $\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$ partitioned into $M=10$ uniform probability bins.

### 5.3 Epistemic Uncertainty & Selective Prediction
Monte Carlo Dropout evaluates $N=10$ stochastic forward passes with active dropout:
$$\mu(x, y) = \frac{1}{N} \sum_{i=1}^N p^{(i)}(x, y), \quad \sigma^2(x, y) = \frac{1}{N} \sum_{i=1}^N (p^{(i)}(x, y) - \mu(x, y))^2$$
- **Selective Prediction:** Evaluate segmentation metrics across retained pixels when masking out the top $k\%$ most uncertain pixels ($\sigma^2 > \tau$). A valid uncertainty metric must demonstrate monotonic accuracy gains as high-uncertainty pixels are excluded.

### 5.4 Robustness & Geographic Generalization
1. **Cloud Degradation Stress Test:** Ingest synthetic cloud contamination in optical imagery at levels $0\%, 25\%, 50\%, 75\%, 100\%$ and plot the performance decay curve.
2. **Missing Modality Sweep:** Evaluate performance when optical, SAR, or DEM is entirely absent.
3. **Out-of-Region Drop:** Compute the percentage reduction in Dice score on the held-out test regions compared to validation regions.
