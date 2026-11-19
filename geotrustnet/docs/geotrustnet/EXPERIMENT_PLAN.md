# GeoTrustNet Experiment Plan

## Evaluation principles
- Fix region/event splits before model comparisons.
- Tune hyperparameters using training/validation only.
- Use the same test split for every model.
- Report per-region metrics as well as aggregate metrics.
- Include multiple seeds where compute permits.
- Report uncertainty intervals or variation across seeds where feasible.
- Include failed runs and negative results in research notes.

## Experiment ladder
| ID | Model | Question |
|---|---|---|
| B0 | Majority/constant or simple heuristic | Is the pipeline/metric implementation sane? |
| B1 | Random Forest / XGBoost on tabular features | Classical susceptibility baseline; only for compatible labels/features |
| B2 | U-Net, single modality | Spatial segmentation baseline |
| B3 | U-TAE or suitable temporal baseline | Does temporal modeling help? |
| B4 | CNN + terrain feature fusion | Does terrain context help? |
| B5 | CNN + temporal Transformer | Does learned temporal attention help? |
| B6 | SAR + optical + terrain naïve fusion | Does multimodality help? |
| P1 | GeoTrustNet reliability-aware fusion | Does learned reliability improve over naïve fusion? |
| P2 | GeoTrustNet + uncertainty method | Are uncertainties informative and calibrated? |

Do not compare models trained on incompatible samples/splits without clearly explaining the difference. The classical tabular baseline belongs primarily to susceptibility unless a valid pixel/sample formulation is defined.

## Core metrics
### Segmentation
- Precision, recall, F1/Dice
- Intersection over Union (IoU)
- PR-AUC (especially under class imbalance)
- ROC-AUC (supplementary; can look optimistic under severe imbalance)
- Confusion matrix and per-region metrics

### Calibration and uncertainty
- Expected Calibration Error (ECE), with binning details
- Brier score
- Negative log-likelihood (NLL)
- Risk-coverage or error-detection curve
- Correlation/association between uncertainty and error

### Robustness
- Missing-modality tests
- Controlled cloud/invalid-pixel degradation for optical data
- Controlled noise/degradation tests with realistic ranges
- Temporal observation dropout
- Geographic/event-held-out test

## Ablations
- remove temporal encoder;
- remove terrain;
- remove SAR;
- remove optical;
- replace reliability fusion with concatenation;
- replace learned reliability with equal weights;
- remove uncertainty estimator;
- compare uncertainty methods if feasible.

## Results ledger
For every run record:
- run_id, timestamp, git commit;
- dataset version and split manifest hash;
- model/config and seed;
- input modalities and missingness;
- best epoch/checkpoint;
- metrics on validation and test;
- hardware/runtime;
- logs and known issues.

## No fabricated targets
Do not promise a particular F1/IoU or claim improvement before experiments. A failed hypothesis is still a valid research result if the evaluation is rigorous.
