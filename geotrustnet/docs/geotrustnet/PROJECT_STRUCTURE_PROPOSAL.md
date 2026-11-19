# Proposed Project Structure

This is a target architecture, not a mandate to reorganize the existing repository. Antigravity must map it onto the current structure after the audit and avoid duplicate modules.

```text
LandSlide/
├── AGENTS.md
├── README.md
├── dashboard/                 # preserve existing Streamlit app/pages
├── data/
│   ├── manifests/
│   ├── processed/
│   └── synthetic/              # tiny fixtures only
├── src/
│   └── geotrustnet/
│       ├── data/
│       │   ├── datasets.py
│       │   ├── preprocessing.py
│       │   ├── geospatial.py
│       │   └── splits.py
│       ├── models/
│       │   ├── encoders.py
│       │   ├── temporal.py
│       │   ├── fusion.py
│       │   ├── decoder.py
│       │   └── geotrustnet.py
│       ├── training/
│       │   ├── engine.py
│       │   └── losses.py
│       ├── evaluation/
│       │   ├── segmentation.py
│       │   ├── calibration.py
│       │   ├── robustness.py
│       │   └── geographic.py
│       └── inference/
│           └── predict.py
├── configs/
│   ├── data/
│   ├── models/
│   └── experiments/
├── scripts/
├── tests/
├── experiments/
├── docs/
│   ├── geotrustnet/
│   └── reports/
├── paper/
├── requirements.txt            # or existing dependency manager
└── .gitignore
```

## Architecture boundaries
- Dashboard imports inference interfaces, not training internals.
- Data loading returns documented tensors plus metadata/validity masks.
- Models do not perform file I/O.
- Training code owns optimization/checkpointing/logging.
- Evaluation code is independent and testable.
- Geospatial inference writes georeferenced outputs with CRS/transform preserved.
