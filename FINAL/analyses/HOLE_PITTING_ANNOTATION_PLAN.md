# Hole/Pitting Detection: Manual Annotation + RF-DETR Plan

## Background

Both classical shot-hole detectors (brightness-only and soil-aware Lab) were manually rejected — see `CONTINUATION_PLAN.md`, "Focused shot-hole improvement". The next defensible step, per supervisor guidance, is a small manually annotated segmentation/detection set, optionally SAM-assisted, with RF-DETR for downstream detection. This is exploratory side work; it does not block or modify the frozen area-weighted MIL model or the 73-image test split.

## Scope

Two classes only: `shot_hole`, `pitting`. Leaf area is already handled by the existing green-mask pipeline and is out of scope here.

## Steps

### 1. Select images — done
`src/preprocessing/collect_hole_pitting_candidates.py` (config: `configs/hole_pitting_candidates.json`) takes the union of the 20 `hole_detection_comparison` images and the 40 `direct_damage_audit` images (40 unique after dedup, all train/val, already stratified), resolves raw paths from the frozen baseline manifest, and copies them to `outputs/hole_pitting_annotations/images/`. It also writes `outputs/hole_pitting_annotations/annotation_candidates.csv` with an 80/20 train/eval split assignment (seed 73). Already run: 32 train, 8 eval images collected.

Rerun with:
```bash
python -m src.preprocessing.collect_hole_pitting_candidates --config configs/hole_pitting_candidates.json
```

### 2. Annotate — manual, not automatable
```bash
pip install anylabeling   # PyPI name; "x-anylabeling" is the GitHub project name, not the pip package
```
This needs a display, so it must be run on your own machine, not this server. Point it at `outputs/hole_pitting_annotations/images/`. Label the raw images with SAM-assisted bounding boxes for `shot_hole` and `pitting` only (leaf area is already handled elsewhere). Export as **COCO**, split into `train/` and `valid/` subfolders under `outputs/hole_pitting_annotations/coco_export/` matching `annotation_candidates.csv`'s `annotation_subset` column (train images → `coco_export/train/`, eval images → `coco_export/valid/`), each with its own `_annotations.coco.json`.

### 3. Fine-tune RF-DETR — scripted, blocked on step 2
`rfdetr` is installed in `.venv`. Run:
```bash
python -m src.training.train_rfdetr_hole_pitting --config configs/rfdetr_hole_pitting_train.json
```
Uses `RFDETRNano`, reads `configs/rfdetr_hole_pitting_train.json` (dataset dir, epochs=20, batch_size=4). Fails fast with a clear message if `coco_export/train/_annotations.coco.json` is missing.

### 4. Evaluate — scripted, blocked on step 2/3
```bash
python -m src.evaluation.evaluate_rfdetr_hole_pitting --config configs/rfdetr_hole_pitting_eval.json
```
Loads the trained checkpoint, runs inference on `coco_export/valid/`, reports mAP@50 and mAP@50:95 via `supervision`, and writes `outputs/rfdetr_hole_pitting/rfdetr_hole_pitting_review.csv` with per-image predicted vs. ground-truth box counts and empty `review_status`/`usable`/`review_notes` columns for manual pass/fail review — the same bar the classical detectors failed (0/40 usable).

### 5. Decide — manual, after step 4

Status: complete. **Rejected.**

40 images were manually annotated in AnyLabeling (36 with visible damage, 342 boxes: 283 train / 59 valid). Full-frame images are ~4000x3000 px while RF-DETR resizes inputs to 384x384; a typical hole/pitting box (~18x18 px) shrank to under 2 px after resize, producing exactly 0.0 mAP through 20 epochs of training. Retraining after tiling images into 640x640 crops around annotation clusters (`src/preprocessing/tile_hole_pitting_coco.py`, 71 train / 16 valid tiles, box-to-image ratio improved ~6x) still failed: validation mAP@50 stayed under 2% throughout 40 epochs, and at the default confidence threshold the model predicts nothing. Lowering the threshold to check for any learned signal produced 100-220 near-random boxes per image with mAP@50 of 0.4% -- not a thresholding issue, the model has not learned to localize holes or pitting.

Most likely cause: 32 training images (283 boxes) is too little data for RF-DETR to converge on tiny, low-contrast targets, even with SAM-assisted annotation quality. This does not contradict the supervisor's "little effort, few examples" expectation in general -- it means this specific dataset size and object scale combination did not reach a usable model within the time available for this side project.

Decision: do not pursue this further. Separate hole/pitting detection remains an open problem alongside the two already-rejected classical CV approaches (`CONTINUATION_PLAN.md` step 4). Report it as exploratory work that did not reach a validated result, per the supervisor's original guidance to prioritize the area-weighted scoring model (option 1) and OOD evaluation as the primary deliverable. The frozen `mil_weighted_seed42` model and its OOD results are unaffected.

Artifacts: `outputs/hole_pitting_annotations/` (candidates, raw annotations, tiled COCO exports), `outputs/rfdetr_hole_pitting/` (checkpoints, `metrics.csv`, `rfdetr_hole_pitting_review.csv`).

## Constraints

- Do not touch the frozen 73-image test split.
- Do not modify or retrain the selected `mil_weighted_seed42` model.
- Keep all outputs under `outputs/rfdetr_hole_pitting/` and annotation exports under a new `outputs/hole_pitting_annotations/`, separate from existing audit artifacts.
