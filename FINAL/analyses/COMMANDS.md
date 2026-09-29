# Complete Command & Execution Guide: CSFB Damage Quantification Pipeline

This document provides a comprehensive command reference for the complete Cabbage Stem Flea Beetle (CSFB) plant phenotyping and damage quantification pipeline. It covers data curation, classical computer vision preprocessing, deep representation learning, multiple instance learning (MIL), out-of-distribution benchmarking, and manual quality audits.

All commands are executed from the project root directory (`FINAL`).

---

## 1. Environment Setup & Dependency Verification

Activate the Python virtual environment and verify essential dependencies.

```bash
# Activate virtual environment
source .venv/bin/activate

# Verify PyTorch, OpenCV, and dependency versions
python -c "import torch, torchvision, cv2, pandas, sklearn; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
```

---

## 2. Dataset Manifest Curation & Split Creation

Commands to build the curated baseline manifest and generate deterministic, leakage-safe train/val/test splits grouped by plot ID (`plot_group`).

```bash
# Build the baseline manifest from raw dataset metadata
python -m src.data.make_baseline_manifest

# Generate plot-group stratified train/val/test splits
python -m src.data.make_baseline_splits

# Generate general dataset manifest
python -m src.data.make_manifest
```

---

## 3. Reference Metal Frame Crop & Alignment

Detects the metal reference frame corners and crops the interior region to remove background soil and un-framed vegetation.

```bash
# Execute metal reference frame cropping audit across 30 sample images
python -m src.preprocessing.frame_crop --config configs/frame_audit.json
```

---

## 4. High-Resolution Plant Instance Detection & Region Proposals

Extracts individual plant crop patches and bounding boxes from inside the metal frame using adaptive HSV color thresholding.

```bash
# Execute plant region proposal extraction audit across 30 sample images
python -m src.preprocessing.plant_regions --config configs/plant_region_audit.json
```

---

## 5. Classical Computer Vision & Biological Feature Extraction (Exploratory Work)

Audits and feature extraction tools for direct leaf area estimation, shot-hole counting, yellow/brown pitting quantification, and soil-aware color segmentation.

### 5.1 Direct Damage Measurement Audit (Leaf Area, Holes & Pitting)
Estimates direct damage percentage by counting enclosed hole pixels and pitting contours inside valid plant regions.

```bash
# Generate direct damage overlays, binary masks, and contact sheets (40 sample images)
python -m src.preprocessing.direct_damage_audit --config configs/direct_damage_audit.json

# Summarize manual review status
python -m src.preprocessing.direct_damage_audit --config configs/direct_damage_audit.json --summarize-review
```

### 5.2 Soil-Aware vs Brightness Hole Classification Comparison
Compares HSV brightness-based hole detection against CIELAB soil-color similarity on 20 difficult field images.

```bash
# Generate side-by-side comparison overlays and high-resolution review pages
python -m src.preprocessing.hole_detection_comparison --config configs/hole_detection_comparison.json

# Summarize manual review conclusion
python -m src.preprocessing.hole_detection_comparison --config configs/hole_detection_comparison.json --summarize-review
```

### 5.3 Plant Mask Pixel Area Weight Audit
Audits plant green mask pixel counts and normalized area weights across 40 sample images.

```bash
# Generate per-image audit overlays and contact sheets
python -m src.preprocessing.plant_weight_audit --config configs/plant_weight_audit.json

# Summarize manual review status
python -m src.preprocessing.plant_weight_audit --config configs/plant_weight_audit.json --summarize-review
```

### 5.4 Morphological Biological Feature Extraction
Extracts biological features (leaf area, hole count, pitting area, perimeter) for plant region crop patches.

```bash
python -m src.preprocessing.biological_features --config configs/biology_audit.json
```

---

## 6. Whole-Image Downsampled Baseline Models

Trains baseline regression models directly on downsampled $224 \times 224$ full field frames (without patch-based plant region extraction).

```bash
# Train whole-image baseline using MSE loss
python main.py --config configs/config_mse.json
```

---

## 7. Deep Representation Learning & Feature Caching

Pre-extracts deep visual feature representations using the frozen DINOv3 backbone model for high-resolution plant patches.

```bash
# Pre-extract and cache DINOv3 patch feature bags to outputs/cache/dinov3_bags.pt
python main.py --config configs/config_mil_data.json
```

---

## 8. Multiple Instance Learning (MIL) Neural Aggregation Pipeline

Trains neural aggregation models (`weighted`, `uniform`, `abmil`, `gated_abmil`) and evaluates in-distribution performance.

### 8.1 Train Production Area-Weighted MIL Model
Trains the Area-Weighted MIL model (`mil_weighted_seed42`) using normalized plant mask area weights.

```bash
python main.py --config configs/config_mil_train.json
```

### 8.2 Evaluate Model on Frozen In-Distribution Test Set
Evaluates predictions on the frozen 73-image test split.

```bash
python main.py evaluate_mil --config configs/config_mil_eval.json
```

### 8.3 12-Run Aggregation Grid Experiment Matrix
Runs the multi-seed grid experiment across 4 aggregation methods and 3 random seeds (42, 43, 44).

```bash
# Execute all 12 experimental runs
python scripts/run_aggregation_experiments.py --config configs/aggregation_experiments.json

# Summarize existing experiment checkpoints without retraining
python scripts/run_aggregation_experiments.py --config configs/aggregation_experiments.json --summarize-existing
```

### 8.4 Fast Pipeline Smoke Test
Runs a fast single-epoch verification run.

```bash
python main.py --config configs/smoke_config.json
```

---

## 9. Zero-Shot Out-of-Distribution (OOD) Field Benchmarks

Evaluates model generalization zero-shot on unseen field trial datasets (`Rauischholzhausen_WG1` — actually **Weilburger Grenze**, folder `WG1`; the name in the code is wrong — and `DSV_Trial_1`), supporting automatic local and server NFS path resolution.

```bash
python main.py --config configs/config_mil_eval.json
```

---

### 9.1 Extended generalization study (used in the report)

Extracts plant bags once for Weilburger Grenze, DSV, held-out GG1 images where the raters disagree,
and the GG1 test split (sanity check), then evaluates all MIL checkpoints with constant baselines,
strata, plot-level metrics and rater agreement. Writes `outputs/tables/generalization/`. ~15 min.

```bash
python scripts/evaluate_generalization.py                 # extract + evaluate
python scripts/evaluate_generalization.py --skip-extract  # re-evaluate from outputs/cache/generalization_bags.pt
```

---

## 9b. Report-specific experiments (2026-09-27)

```bash
# Ranking-based training, 3 seeds (joint Huber + margin ranking), ~1 min on cached features
python scripts/train_joint_ranking.py

# Whole-image baseline reproduction (Huber, seed 42) and its test evaluation
python main.py train --config configs/config_mse.json --loss huber --run_name wholeimage_huber_seed42 \
  --out_manifest outputs/tables/baseline_manifest_split.csv --image_size 224 --high_quality_only False
python main.py evaluate --config configs/config_mse.json --out_manifest outputs/tables/baseline_manifest_split.csv \
  --model_path outputs/runs/wholeimage_huber_seed42/checkpoints/best_model.pth --image_size 224

# Predicted-vs-true scatter figure for the report (needs 9.1 first)
python scripts/plot_report_scatter.py

# Build the report (or save in VS Code with LaTeX Workshop)
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---

## 9c. RF-DETR Hole/Pitting Detector (pretrained weights)

The trained weights (`checkpoint_best_total.pth`, 116 MB) are not tracked in git. Download them from Google Drive:
https://drive.google.com/drive/folders/15my71BUyC346ksATji6mAB_NKftLYRYn?usp=share_link

Place the file at `outputs/rfdetr_hole_pitting/checkpoint_best_total.pth` (this is the default path used by the scripts and `configs/rfdetr_hole_pitting_eval.json`).

```bash
pip install rfdetr
mkdir -p outputs/rfdetr_hole_pitting
mv ~/Downloads/checkpoint_best_total.pth outputs/rfdetr_hole_pitting/

# Render labeled/predicted example figures
python scripts/render_rfdetr_hole_pitting_examples.py
```

Use in Python (the model must be loaded as `RFDETRNano`, the size it was trained with):

```python
from rfdetr import RFDETRNano
from PIL import Image

model = RFDETRNano(pretrain_weights="outputs/rfdetr_hole_pitting/checkpoint_best_total.pth")
dets = model.predict(Image.open("image.jpg"), threshold=0.5)
print(dets.xyxy, dets.class_id, dets.confidence)
```

---

## 10. Automated Test Suite (pytest)

Run automated unit tests to verify data contracts, configuration parsing, preprocessing functions, and model architectures.

```bash
# Run the complete test suite
pytest

# Run tests with verbose output
pytest -v

# Run specific test modules
pytest tests/test_config.py
pytest tests/test_frame_crop.py
pytest tests/test_plant_regions.py
pytest tests/test_biological_features.py
pytest tests/test_direct_damage_audit.py
pytest tests/test_hole_detection_comparison.py
pytest tests/test_plant_weight_audit.py
pytest tests/test_mil_model.py
pytest tests/test_mil_dataset.py
pytest tests/test_aggregation_experiments.py
```
