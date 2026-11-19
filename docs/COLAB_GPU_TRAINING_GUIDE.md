# Google Colab / Cloud GPU Training Guide — GeoTrustNet U-Net Baseline

This guide details how to execute GPU-accelerated training of the GeoTrustNet Sentinel-1 SAR baseline segmentation model using Google Colab or Kaggle Notebooks.

---

## 1. Environment Setup

### In Google Colab
1. Navigate to **Edit > Notebook Settings > Hardware accelerator** and select **T4 GPU** (or A100/V100 if available).
2. Clone or upload your repository to Colab:
   ```bash
   !git clone <YOUR_REPOSITORY_URL> landslide
   %cd landslide
   ```
3. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
4. Verify GPU availability:
   ```python
   import torch
   print("CUDA Available:", torch.cuda.is_available())
   print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
   ```

---

## 2. Dataset Preparation

If working with synthetic fixtures for integration verification:
```bash
python scripts/generate_fixtures_and_manifests.py
```

If working with full Sen12Landslides patches:
1. Place or mount dataset patches under `data/sen12landslides/`.
2. Generate the leakage-safe split manifest:
   ```bash
   python scripts/generate_fixtures_and_manifests.py --manifest data/manifests/sen12landslides_manifest.csv
   ```

---

## 3. Running Baseline Training

Execute the baseline training script with GPU auto-detection:
```bash
python scripts/train_unet_baseline.py \
    --config configs/experiments/baseline_s1_experiment.json \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001
```

### Key Training Parameters
- `--config`: Experiment JSON definition (`configs/experiments/baseline_s1_experiment.json`).
- `--epochs`: Number of training epochs (e.g., 50 with early stopping patience of 10).
- `--batch-size`: Batch size (e.g. 16 on T4 GPU 16GB VRAM, 32 on A100 40GB).
- `--device`: Target device (`cuda` or `cpu`).

---

## 4. Experiment Outputs & Artifacts

All outputs are saved to `artifacts/geotrustnet/experiments/baseline_s1/`:
- `best_model.pt`: Weights at epoch with maximum validation IoU.
- `latest_checkpoint.pt`: Checkpoint of the latest completed epoch for resuming.
- `training_history.json`: Per-epoch losses, learning rates, and validation metrics (`iou`, `dice`, `precision`, `recall`, `pr_auc`).
- `experiment_summary.json`: Git commit hash, hardware device, best metrics, duration, and parameter configuration.
- `test_evaluation_report.json`: Final objective evaluation on the held-out test split.

---

## 5. Research Integrity Protocols
- The final model is selected strictly on the **validation set** (`val_iou`).
- The **test set** is evaluated once at the very end to prevent data leakage and hyperparameter overfitting.
- Synthetic fixtures must never be reported as real-world benchmark metrics.
