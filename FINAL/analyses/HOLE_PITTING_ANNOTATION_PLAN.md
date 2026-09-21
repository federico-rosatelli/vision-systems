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

Status: complete. **Revised 2026-09-21 after finding and fixing a tiling bug — weak positive result, not a clean rejection.**

40 images were manually annotated in AnyLabeling (36 with visible damage, 342 boxes: 283 train / 59 valid). Full-frame images are ~4000x3000 px while RF-DETR resizes inputs to 384x384; a typical hole/pitting box (~18x18 px) shrank to under 2 px after resize, producing exactly 0.0 mAP through 20 epochs of training. Retraining after tiling images into 640x640 crops around annotation clusters (`src/preprocessing/tile_hole_pitting_coco.py`, 71 train / 16 valid tiles, box-to-image ratio improved ~6x) still originally failed: validation mAP@50 stayed under 2% throughout 40 epochs, and at the default confidence threshold the model predicted nothing. Lowering the threshold produced 100-220 near-random boxes per image with mAP@50 of 0.4%.

**Root cause found (2026-09-21), not "too little data":** all 40 annotated source images carry a non-default EXIF orientation tag (36 at 180 deg, 2 at 90 deg). `tile_hole_pitting_coco.py` read images with `PIL.Image.open()`, which auto-applies EXIF orientation on load in this Pillow version (12.3), while the annotation bbox coordinates (from AnyLabeling) are in the raw, un-rotated pixel frame — the same frame `cv2.imread()` returns, and the frame the rest of the codebase already standardizes on via `read_image_oriented()` in `frame_crop.py`. The tiling script cropped the rotated image but placed boxes using un-rotated coordinates, so essentially every box in the tiled train/eval set was spatially decorrelated from its image content — visually confirmed on multiple examples (ground-truth boxes landing squarely on soil or the plot frame, nowhere near a leaf). This alone is sufficient to produce near-zero mAP and near-random predictions regardless of dataset size. Only this tiling path was affected; the frozen `mil_weighted_seed42` model, classical CV audits, and all other pipelines use the EXIF-aware reader and are unaffected.

Fix: `tile_hole_pitting_coco.py` now uses `cv2` for all image I/O instead of `PIL.Image`, matching the raw pixel frame the annotations are defined in (verified visually against both the 180-deg and 90-deg cases). The tiled dataset was regenerated (71 train / 16 valid tiles, same counts as before) and RF-DETR was retrained from scratch on the corrected tiles.

**Result after the fix:** validation mAP@50 improved roughly 20-25x, from 0.4% to a peak of 10.0% (epoch 24) over the 40-epoch run; the official eval script (`checkpoint_best_total.pth`, confidence threshold 0.5) reports **mAP@50 = 3.47%, mAP@50:95 = 1.65%** (previously 0.39% / 0.18%). Per-image predicted box counts are now sane (0-3 per tile, matching plausible damage counts) instead of 100-300 near-random boxes. Qualitatively, predicted boxes now cluster on visible leaf damage rather than on soil texture or the plot frame — see `outputs/rfdetr_hole_pitting/example_figures/` for before/after comparisons (`example_figures_OLD_BUGGY/` keeps the pre-fix renders for reference).

This is still a weak detector (mAP@50 ~3-10%, well short of production usability) and 32 training images / 283 boxes for ~18px objects remains a genuine, separate limiting factor on top of the now-fixed labeling bug — the original "too little data" concern is real, just not the primary explanation for the near-zero result that was reported before.

Decision: report this as a partially-successful, still-exploratory result, not a clean rejection: a real (if weak) detection signal exists once labels are correct, but the dataset is too small to make it production-useful within this project's scope. The bug and its fix are worth reporting explicitly as a methodology lesson (garbage-in-garbage-out negative results should be checked for pipeline bugs before being attributed to data scarcity). The frozen `mil_weighted_seed42` model and its OOD results remain unaffected and are still the primary deliverable.

Artifacts: `outputs/hole_pitting_annotations/` (candidates, raw annotations, tiled COCO exports; pre-fix buggy tiles preserved at `coco_export_tiled.OLD_BUGGY/`), `outputs/rfdetr_hole_pitting/` (checkpoints, `metrics.csv`, `rfdetr_hole_pitting_review.csv`, `example_figures/`; pre-fix outputs preserved at `outputs/rfdetr_hole_pitting.OLD_BUGGY/`).

### Example figures (post-fix)

Four `outputs/rfdetr_hole_pitting/example_figures/*_gt_vs_pred.jpg` files (side-by-side
ground truth vs. prediction at confidence threshold 0.5), generated with
`scripts/render_rfdetr_hole_pitting_examples.py`, chosen to show the range of outcomes
rather than only the best case:

| Tile | GT boxes | Predicted boxes | What it shows |
| :--- | :---: | :---: | :--- |
| `20251021_122655_11` | 5 | 3 | Best case: predictions land on real damage, 2 of 5 missed. |
| `20251021_151409_9` | 4 | 2 | Partial: the two damage marks on an isolated leaflet are missed; a nearby 2-mark cluster is caught. |
| `20251021_132633_1` | 1 | 0 | Miss: a single isolated hole produces no detection at threshold 0.5. |
| `20251021_122353_12` | 30 | 0 | Worst case: a dense row of seedlings with 30 marks produces zero detections. |

Pattern across all four: the model no longer hallucinates boxes on soil/frame texture
(a categorical improvement over the pre-fix model), but it is under-confident and
recall-limited — it detects roughly 40-60% of marks in small clusters and misses
isolated or very dense damage entirely. This is consistent with a model that has
learned a real but weak signal from too few examples, particularly for the
minority `pitting` class (see below).

### How many annotated images would this need?

Current annotated set: 40 images (32 train / 8 valid), 342 boxes across two
imbalanced classes:

| Class | Train boxes | Valid boxes |
| :--- | :---: | :---: |
| `shot_hole` | 222 | 42 |
| `pitting` | 61 | 17 |

`pitting` is the more data-starved class (3.6x fewer train boxes than
`shot_hole`), which likely drags overall mAP down further.

Rough estimate, based on general small-object detection practice (targets here
are ~18px, low-contrast against soil) rather than a formal scaling-law fit:

- **~150-300 annotated images with damage per class** (roughly 5-10x the
  current count, ~1,500-3,000 boxes total, kept balanced between `shot_hole`
  and `pitting`) to reach a "usable for triage" detector (mAP@50 in the
  ~30-50% range).
- **500+ images per class** to reach something reliable enough to use
  unsupervised (mAP@50 ~70%+).

This is a rough estimate, not a guarantee — annotation consistency and how
visually separable `pitting` really is from soil texture (even to a human)
both affect the real data efficiency. Getting to 150-300 well-annotated
images would be a substantial additional annotation effort on top of the 40
already done, and is out of scope for this project's remaining time; noted
here as the concrete next step if this direction were resumed later.

## Constraints

- Do not touch the frozen 73-image test split.
- Do not modify or retrain the selected `mil_weighted_seed42` model.
- Keep all outputs under `outputs/rfdetr_hole_pitting/` and annotation exports under a new `outputs/hole_pitting_annotations/`, separate from existing audit artifacts.
