# GeoTrustNet Model Design

## Design philosophy
Build incrementally. Keep a working baseline at every stage. Each new module must have a test, configuration option, and measurable ablation.

## Proposed components
### 1. SAR branch
Input: available Sentinel-1 channels (for example VV/VH), validity mask, and acquisition metadata.
Encoder: compact CNN/ResNet.
Output: spatial feature map(s).

### 2. Optical branch
Input: available Sentinel-2 bands and validity/cloud masks.
Encoder: compact CNN/ResNet initially. Consider ViT only after a baseline and adequate training data/compute are established.
Output: spatial feature map(s).

### 3. Temporal module
Input: ordered per-date feature embeddings and time metadata.
Candidate: temporal attention/Transformer. Compare with a simpler temporal pooling or ConvLSTM baseline.
Must support variable/missing observation counts through explicit masks.

### 4. Terrain branch
Input: aligned DEM-derived raster layers or tabular terrain features.
Use a CNN for raster context or an MLP for tabular variables. Do not mix these representations without an explicit interface.

### 5. Fusion
Implement in this order:
1. single-modality baselines;
2. feature concatenation;
3. gated/equal-weight fusion;
4. reliability-aware fusion;
5. optional cross-attention if justified by validation results.

Fusion must handle missing modalities without NaNs and expose modality availability/weights for diagnostics.

### 6. Segmentation decoder
U-Net-style decoder with skip connections and output resolution matching the target mask. Use logits during training and sigmoid/softmax only for inference/metrics as appropriate.

### 7. Uncertainty
Start with one method, preferably MC Dropout or a small deep ensemble depending on compute.
- Predictive entropy and/or variance maps.
- Evaluate uncertainty against errors.
- Calibration metrics: ECE, Brier score, NLL.
- Clearly distinguish confidence from correctness.
- Do not display a raw softmax score as a calibrated confidence unless calibration has been measured.

## Suggested modular interfaces
Adapt names to repository conventions after audit:
- `data/datasets.py`: sample loading and metadata.
- `data/transforms.py`: deterministic/geospatial transforms.
- `models/encoders.py`: modality encoders.
- `models/temporal.py`: temporal modeling.
- `models/fusion.py`: concatenation and reliability-aware fusion.
- `models/decoder.py`: segmentation decoder.
- `models/geotrustnet.py`: top-level model.
- `training/engine.py`: training loop.
- `evaluation/segmentation.py`: segmentation metrics.
- `evaluation/calibration.py`: calibration/uncertainty metrics.
- `evaluation/geographic.py`: grouped spatial evaluation.
- `inference/predict.py`: inference and geospatial output writing.

## Shape and contract tests
Write tests for:
- each encoder input/output shape;
- temporal masks and variable observation lengths;
- missing SAR/optical/terrain modalities;
- fusion weights sum to one over available modalities;
- all-missing input fails clearly;
- segmentation logits match target spatial size;
- uncertainty output shape and finite values;
- checkpoint save/load;
- CPU forward-pass smoke test.

## Training requirements
- Seed Python, NumPy, and PyTorch.
- Log dataset/split/config/code revision.
- Save best checkpoint based on a declared validation metric.
- Use early stopping only with a documented criterion.
- Track class imbalance and choose loss (BCE/Dice/Focal or combination) based on validation, not test data.
- Never tune on final held-out test geography.
