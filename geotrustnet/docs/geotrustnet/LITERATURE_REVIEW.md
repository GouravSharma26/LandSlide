# Literature Review Register

## Purpose
Maintain a verifiable literature review. Do not cite papers from memory alone. Record exact title, authors, venue, year, DOI/URL, task, dataset, split, metrics, and relevance.

## Seed paper: Sen12Landslides
Höhn et al., “A Spatio-Temporal Dataset for Satellite-Based Landslide Detection,” *Scientific Data* (2025).
- DOI: https://doi.org/10.1038/s41597-025-06167-2
- Article: https://www.nature.com/articles/s41597-025-06167-2
- Code: https://github.com/PaulH97/Sen12Landslides
- Dataset: https://huggingface.co/datasets/paulhoehn/Sen12Landslides

Questions to extract from the paper and supplement:
- exact sample/patch counts by modality and region;
- which samples are spatially aligned across modalities;
- temporal observation structure;
- label-generation/refinement process;
- official train/validation/test protocol;
- exact baseline metrics and table/figure references;
- limitations and known failure modes.

## U-TAE reference
Garnot and Landrieu, “Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks,” ICCV 2021.
Find and verify the official paper, implementation, input format, and benchmark before reproducing.

## Review table template
| Paper | Task | Sensors/data | Split strategy | Model | Metrics/results | Limitation | Relevance to GeoTrustNet |
|---|---|---|---|---|---|---|---|

## Required review themes
1. Satellite landslide detection and segmentation.
2. Sentinel-1 SAR and Sentinel-2 optical fusion.
3. Temporal satellite image modeling.
4. Multimodal reliability/quality-aware fusion.
5. Uncertainty estimation and calibration in segmentation.
6. Geographic domain generalization and spatial leakage.
7. Landslide susceptibility versus event detection versus forecasting.

## Research-gap statement template
“Prior work has established ______. However, under ______ evaluation conditions, ______ remains insufficiently addressed. We investigate ______ using ______, and evaluate it through ______.”

Complete this only after reviewing and comparing primary sources. Do not state that no prior work exists without a documented search.
