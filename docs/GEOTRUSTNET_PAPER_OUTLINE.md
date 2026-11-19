# GeoTrustNet: Uncertainty-Aware Spatio-Temporal Multimodal Deep Learning for Reliable Landslide Detection and Geographic Generalization

**IEEE Transactions on Geoscience and Remote Sensing (TGRS) / Nature Scientific Data Format**  
**Working Paper Outline & Publication Plan**  

---

## Section 1: Abstract
- **Context:** Satellite remote sensing is critical for landslide mapping and disaster response in steep, inaccessible mountain ranges. However, environmental factors—monsoon cloud cover obscuring optical satellites, radar geometric distortions in complex topography, and severe class imbalance—render conventional deep-learning models brittle and uncalibrated.
- **Problem Statement:** Existing multimodal fusion approaches rely on static concatenation or unweighted summation, propagating corrupted features from degraded sensors into downstream predictions without flagging prediction uncertainty.
- **Proposed Solution:** We present **GeoTrustNet**, an uncertainty-aware multimodal segmentation architecture featuring:
  1. Independent sensor encoders preserving domain-specific hierarchies for Sentinel-1 SAR, Sentinel-2 Optical, and Copernicus DEM.
  2. Multi-scale spatial reliability estimation that dynamically reweights modality representations based on localized observational quality.
  3. Epistemic uncertainty quantification via Monte Carlo Dropout, enabling selective prediction (error-rejection) for high-stakes emergency management.
- **Evaluation Framework:** Evaluated using a strict disjoint regional holdout protocol on the Sen12Landslides benchmark dataset to test true geographic generalization across unseen mountain terrains.
- **Significance:** Demonstrates a path toward trustworthy, reliable remote sensing intelligence where deep learning systems quantify their own doubt rather than producing overconfident, uncalibrated hazard maps.

---

## Section 2: Introduction
- **2.1 Operational Challenges in Mountainous Landslide Detection:**
  - High vulnerability of Himalayan and Andean communities to rainfall-triggered landslides.
  - Failure modes of unimodal Earth Observation: Optical blindspots during intense cloudbursts vs. SAR layover, foreshortening, and shadow.
- **2.2 Limitations of Conventional Multimodal Fusion:**
  - Equal-trust assumption in deep feature concatenation.
  - Absence of sensor degradation awareness.
  - Overconfidence in neural network probability outputs under distribution shift.
- **2.3 The Core Research Hypothesis:**
  - Dynamic spatial reliability weighting significantly improves segmentation robustness under cloud contamination and missing modalities compared to static fusion baselines.
- **2.4 Summary of Key Contributions:**
  - Formulation of adaptive spatial reliability gating for multi-sensor Earth Observation.
  - Dual-framework integrating modality reliability attenuation with epistemic uncertainty.
  - Standardized benchmark framework comparing single-modality SAR, temporal ConvLSTM, conventional fusion, and GeoTrustNet under strict geographic holdouts.

---

## Section 3: Related Work
- **3.1 Deep Learning for Landslide Segmentation:**
  - Conventional U-Net and ResUNet architectures on optical imagery.
  - SAR-based change detection and coherence tracking.
  - Review of existing benchmarks and their spatial autocorrelation pitfalls.
- **3.2 Multimodal Fusion in Remote Sensing:**
  - Early fusion (pixel-level stacking), intermediate fusion (feature concatenation), and late fusion (decision voting).
  - Attention-based fusion and cross-modal transformers.
  - Vulnerability of existing models to missing modalities during deployment.
- **3.3 Uncertainty Quantification in Remote Sensing:**
  - Aleatoric vs. epistemic uncertainty in geospatial deep learning.
  - Monte Carlo Dropout and deep ensembles.
  - Probability calibration (Expected Calibration Error, Brier Score, Platt scaling).

---

## Section 4: Dataset and Preprocessing
- **4.1 Sen12Landslides Benchmark Overview:**
  - Reference: Höhn et al., *Nature Scientific Data* (2025).
  - Geographic distribution across global mountainous regions.
  - Target mask definition and expert polygon delineation.
- **4.2 Multi-Sensor Modalities:**
  - *Sentinel-1 SAR:* Dual-polarization (VV, VH) C-band GRD in decibels (dB).
  - *Sentinel-2 Optical:* 10 surface reflectance spectral bands (B02 to B12) at 10m/20m resolution.
  - *Copernicus DEM:* 30m GLO-30 DSM providing continuous elevation and derived terrain slope.
- **4.3 Preprocessing Standardization:**
  - Dynamic range clipping, linear normalization, and coordinate alignment.
  - Quality mask handling (Scene Classification Layer cloud masking).
  - Nodata handling and spatial resolution harmonization to 10m.

---

## Section 5: Methodology
- **5.1 Architecture Overview:**
  - End-to-end pipeline architecture diagram.
- **5.2 Independent Modality Encoders:**
  - SAR Encoder ($E_{\text{sar}}$): Spatial convolution with microwave speckle filtering.
  - Optical Encoder ($E_{\text{opt}}$): Multi-spectral feature extraction with wide receptive field.
  - Terrain Encoder ($E_{\text{dem}}$): Morphometric slope and elevation gradient encoding.
- **5.3 Adaptive Spatial Reliability Fusion:**
  - Gating network formulation: $S(x, y) = \text{Conv}(\text{Concat}(F_{\text{sar}}, F_{\text{opt}}, F_{\text{dem}}))$.
  - Masked spatial softmax normalization: $w_m(x, y) = \frac{\exp(S_m(x, y) + M_m)}{\sum_{j=1}^3 \exp(S_j(x, y) + M_j)}$.
  - Modality feature reweighting and residual skip connection.
- **5.4 Decoder & Output Heads:**
  - Multi-scale skip fusion into expansive decoder path.
  - Logit generation and spatial upsampling.
- **5.5 Epistemic Uncertainty Estimation:**
  - Monte Carlo Dropout sampling ($N$ forward passes): $\mu(x, y)$, $\sigma^2(x, y)$, and information entropy $\mathcal{H}(x, y)$.
- **5.6 Objective Function:**
  - Combined Binary Cross-Entropy with positive prevalence class weighting + Soft Dice Loss:
    $$\mathcal{L} = \alpha \mathcal{L}_{\text{BCE}}(y, \hat{p}) + (1-\alpha) \mathcal{L}_{\text{Dice}}(y, \hat{p})$$

---

## Section 6: Experimental Protocol
- **6.1 Leakage-Free Regional Holdout:**
  - Strict disjoint partitioning by `region_id` and `event_id` (70% Train, 15% Val, 15% Test).
  - Deterministic random seed controls and manifest hashing.
- **6.2 Controlled Baselines:**
  - Model A: SAR U-Net (Single-date baseline).
  - Model B: Temporal SAR ConvLSTM (Temporal baseline).
  - Model C: Conventional Multimodal Fusion (Static concatenation).
  - Model D: GeoTrustNet (Proposed reliability-aware fusion).
- **6.3 Evaluation Metrics:**
  - Segmentation: Precision, Recall, F1, Dice, IoU, PR-AUC.
  - Calibration: Brier Score, Expected Calibration Error (ECE), Reliability Diagrams.
  - Selective Prediction: Risk-coverage curves and accuracy improvements as high-uncertainty pixels are rejected.

---

## Section 7: Results (Pending Cloud GPU Execution)
- **7.1 Overall Benchmark Performance:**
  - Comparative table reporting Precision, Recall, Dice, IoU, PR-AUC across Models A, B, C, D on the held-out test split.
- **7.2 Segmentation Quality Across Terrain Contexts:**
  - Performance stratified by slope angle (gentle $< 15^\circ$, moderate $15^\circ - 35^\circ$, steep $> 35^\circ$).
- **7.3 Visual Comparison of Scarp Delineation:**
  - Qualitative comparison of scarp boundary delineation across competing models.

---

## Section 8: Controlled Ablation Studies
- **8.1 Impact of Reliability Gating:**
  - Direct comparison of Model D (reliability-aware) vs Model C (fixed concatenation) under identical encoders and training budgets.
- **8.2 Modality Contribution Analysis:**
  - Performance ablation when omitting Optical (simulating cloudy monsoon conditions), SAR (radar failure), or DEM (flat terrain assumption).
- **8.3 Reliability Map Behavior Under Simulated Degradation:**
  - Inspection of spatial weight maps $w_m(x, y)$ as optical bands are artificially obscured with cloud noise ($0\%$ to $100\%$).

---

## Section 9: Geographic Generalization
- **9.1 In-Region vs. Out-of-Region Performance:**
  - Quantification of the generalization gap ($\Delta \text{Dice}$) when evaluating on completely unseen mountain ranges.
- **9.2 Terrain and Climate Shift Analysis:**
  - Robustness to differing geologic and vegetative regimes between train and test regions.

---

## Section 10: Uncertainty and Reliability Analysis
- **10.1 Calibration Assessment:**
  - Analysis of Expected Calibration Error and 10-bin reliability diagrams.
- **10.2 Spatial Uncertainty Distribution:**
  - Correlation between high epistemic variance $\sigma^2(x, y)$ and true prediction error boundaries / scarp fringes.
- **10.3 Selective Prediction for Decision-Makers:**
  - Error-rejection curve demonstrating increased operational precision when abstaining on uncertain pixels.

---

## Section 11: Discussion
- **11.1 Interpretability of Learned Reliability Weights:**
  - Physical interpretation of where and why the network routes feature authority between SAR and Optical sensors.
- **11.2 Implications for Operational Disaster Management:**
  - Using uncertainty maps for prioritization of drone flyovers and field reconnaissance.

---

## Section 12: Research Limitations
- **12.1 Resolution Bottlenecks:**
  - 10m Sentinel resolution limits detection of small shallow landslides ($< 100 \text{ m}^2$).
- **12.2 SAR Temporal Latency:**
  - 6 to 12-day Sentinel-1 revisit cycle introduces temporal lag relative to instantaneous landslide triggering.
- **12.3 Computational Footprint:**
  - Monte Carlo Dropout requires $N$ forward passes during inference, impacting real-time throughput unless accelerated.

---

## Section 13: Conclusion and Future Directions
- **13.1 Summary of Findings:**
  - Recapitulation of research hypothesis verification and reliability-aware multimodal advantages.
- **13.2 Future Work:**
  - Integration of commercial high-resolution imagery (PlanetScope 3m).
  - Integration of precipitation reanalysis priors (GPM IMERG / ERA5) directly into the dynamic gating network.
  - Deployment into interactive cloud-native early warning systems.
