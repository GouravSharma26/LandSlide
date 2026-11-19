# Research Specification

## 1. Research problem
Satellite-based landslide detection is challenging because observations vary over time, sensors have different characteristics, optical data may be obscured by clouds, SAR can be noisy, labels may be uncertain, and models may generalize poorly to geographically unseen regions.

## 2. Core research question
Can a multimodal model that learns temporal context and input reliability improve landslide segmentation robustness and geographic generalization compared with conventional single-modality and naïve-fusion baselines?

## 3. Research objectives
- Establish reproducible segmentation baselines.
- Evaluate the contribution of SAR, optical imagery, and terrain data.
- Model temporal information when repeated observations are available.
- Develop a reliability-aware fusion mechanism for missing/noisy modalities.
- Quantify predictive uncertainty and calibration.
- Evaluate generalization to held-out geographic regions/events.

## 4. Hypotheses (to be tested, not assumed)
H1. Multimodal input can improve segmentation over a suitable single-modality baseline.
H2. Temporal modeling can improve detection when temporal observations contain useful pre-/post-event information.
H3. Reliability-aware fusion can improve robustness when a modality is degraded or missing.
H4. Uncertainty estimates can identify a meaningful subset of errors and support calibrated confidence.
H5. Geographic holdout performance may be lower than random-split performance; this gap should be measured and reported.

## 5. Task definitions
### Task A — event detection / segmentation
Input: spatially aligned satellite observations and optional terrain/context.
Target: binary or multi-class pixel mask for a defined landslide event/inventory.
Output: per-pixel probability and mask.

### Task B — susceptibility
Input: terrain, geology, land cover, rainfall/climate and other static or aggregated predictors.
Target: inventory-derived susceptibility labels with a clearly documented sampling strategy.
Output: spatial susceptibility score/map.

These tasks require separate datasets/labels, train-test splits, metrics, and claims.

### Task C — forecasting (later phase only)
Input: information available strictly before a prediction cutoff.
Target: landslide occurrence in a specified future time window.
This requires timestamped event outcomes, temporal validation, and careful prevention of future-data leakage. Do not label retrospective detection as forecasting.

## 6. Proposed GeoTrustNet architecture
- SAR encoder: CNN/ResNet feature extractor for Sentinel-1 channels such as VV/VH, subject to actual data schema.
- Optical encoder: CNN/ResNet or a compact vision transformer for Sentinel-2 bands, subject to actual band availability.
- Temporal encoder: temporal attention/Transformer or ConvLSTM over timestamped observations.
- Terrain encoder: MLP for tabular terrain variables or CNN for aligned raster layers.
- Fusion: concatenate/gated fusion as a baseline, then reliability-aware attention/gating.
- Segmentation decoder: U-Net-style multi-scale decoder.
- Uncertainty: MC Dropout, deep ensembles, evidential methods, or calibrated probability outputs; select one tractable primary method and compare appropriately.

Do not implement every architecture option simultaneously. Start with the simplest reproducible baseline and add components through ablations.

## 7. Reliability-aware fusion concept
For modality feature tensors \(F_m\), estimate a quality/reliability score \(r_m\) for each available modality. Normalize scores over available modalities and combine features:
\[
\alpha_m = \frac{\exp(r_m)}{\sum_{j \in A}\exp(r_j)}, \qquad
F = \sum_{m \in A}\alpha_m F_m
\]
where \(A\) is the set of available modalities. This is a conceptual starting point. The implementation must define how reliability is learned or supervised, how missing modalities are masked, and how the system avoids learning spurious shortcuts.

Possible quality signals include valid-pixel fraction, cloud/invalid-data masks, SAR noise/quality indicators, and learned feature-level confidence. Compare against simple concatenation and equal-weight fusion.

## 8. Novelty discipline
The architecture components (CNN, Transformer, cross-attention, MC Dropout, Grad-CAM, SHAP) are established techniques. Novelty must come from a precise, literature-supported problem formulation, method design, dataset protocol, or empirical finding. Conduct a current literature review before making novelty claims. Record related work, differences, and limitations in `LITERATURE_REVIEW.md`.

## 9. Primary success criteria
- Correct and reproducible data pipeline.
- Leakage-safe region/event holdout.
- Strong and fairly tuned baselines.
- Complete ablation and robustness studies.
- Calibration/uncertainty evaluation.
- Transparent reporting of negative as well as positive results.
- Reproducible checkpoints/configs and clear limitations.

No target score is guaranteed in advance.
