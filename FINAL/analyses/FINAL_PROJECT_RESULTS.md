# Final Project Results: Cabbage Stem Flea Beetle (CSFB) Leaf Damage Quantification

This document compiles the complete benchmark results, empirical tables, model selection audits, and preprocessing evaluations for the Cabbage Stem Flea Beetle (CSFB) plant damage quantification project.

---

## 1. Executive Summary & Objective

The primary objective is to build a **reproducible computer vision pipeline** that quantifies CSFB feeding damage on oilseed rape (*Brassica napus*) plants within a standardized reference frame. The target metrics are the percentage of visible leaf area affected by **shot holes** and **yellow/brown pitting** for BBCH10 growth stage field images, supporting downstream genotype resistance analysis.

---

## 2. Data Contract & Fixed Dataset Splits

All in-domain development, model selection, and validation strictly adhere to the leakage-safe manifest `outputs/tables/baseline_manifest_split.csv`.

| Dataset Attribute | Value | Verification & Integrity Safeguards |
| :--- | :---: | :--- |
| **Total Verified Population (BBCH10)** | **470 images** | Unique physical images with consensus JLU/GAU expert scores |
| **Training Split (`train`)** | **331 images** | Zero plot-group (`plot_group`) overlap with val/test |
| **Validation Split (`val`)** | **66 images** | Used exclusively for model selection and early stopping |
| **Frozen Test Split (`test`)** | **73 images** | **Kept strictly closed** during model optimization |
| **Cache Manifest Hash (SHA-256)** | Verified | Recorded in `outputs/cache/dinov3_bags.pt` |

---

## 3. Preprocessing & Crop Pipeline Audits

Before modeling, the preprocessing pipeline (metal reference frame detection and plant region proposal) was manually audited on a stratified sample of 30 train/validation images.

| Preprocessing Module | Sample Size | Automatic Success | Manual Approval Rate | Key Audit Metrics |
| :--- | :---: | :---: | :---: | :--- |
| **Metal Reference Frame Detection** | 30 images | 30/30 (100%) | **100.0% (30/30)** | Mean frame confidence: 0.8123 (Reviewed by Emrullah) |
| **Plant Region Proposal Audit** | 30 images | 30/30 (100%) | **100.0% (30/30)** | Extracted 191 plant regions (Median: 6.0 plants/image) |

---

## 4. Whole-Image DINOv3 Baseline Performance

Evaluation of the whole-image baseline model, where complete field frames are downsampled directly to $224 \times 224$ pixels without high-resolution patch extraction.

| Evaluation Metric | Validation Set ($N=66$) | Frozen Test Set ($N=73$) |
| :--- | :---: | :---: |
| **MAE (Mean Absolute Error %)** | 5.1265 | 4.6502 |
| **Pearson Correlation ($r$)** | 0.2391 | -0.0971 |
| **Spearman Rank Correlation ($\rho$)** | 0.2712 | -0.1078 |

Reproduced exactly on 2026-09-27 by retraining from the frozen manifest with Huber loss, seed 42
(`outputs/runs/wholeimage_huber_seed42`): val MAE 5.1265, test MAE 4.6502, RMSE 6.2596, Pearson
-0.0971, Spearman -0.1078, pairwise gap-5 accuracy 0.46 (chance). The original run was therefore the
Huber variant. Constant training median gives test MAE 4.01, so the baseline is worse than a constant.

> **Key Finding**: Downsampling whole images discards small shot-hole details, leading to negative test correlations. This strongly justified moving to a Multiple Instance Learning (MIL) patch-based architecture.

---

## 5. Controlled MIL Aggregation Experiment (3-Seed Audit Across 12 Runs)

Across-seed validation performance (seeds 42, 43, 44) comparing plant-score aggregation functions on `baseline_manifest_split.csv`.

| MIL Aggregation Function | Validation MAE (Mean ± SD) | Spearman Rank $\rho$ (Mean ± SD) |
| :--- | :---: | :---: |
| **Area-Weighted Plant Scores** ($w_i = A_i / \sum A_j$) | **2.6932 ± 0.0232** | **0.8532 ± 0.0022** |
| **Uniform Plant Scores** (Unweighted Mean) | 2.7293 ± 0.0401 | 0.8381 ± 0.0064 |
| **ABMIL** (Attention MIL - Ilse et al., 2018) | 2.7430 ± 0.0307 | 0.8093 ± 0.0109 |
| **Gated ABMIL** (Gated Attention MIL) | 2.7808 ± 0.0243 | 0.8046 ± 0.0107 |

---

## 6. Selected Production Model Performance (`aggregation_weighted_seed42`)

Correction (2026-09-21): this section previously attributed these numbers to
`mil_weighted_seed42`. They are in fact the **validation-set** metrics of
`aggregation_weighted_seed42`, the run selected by the 3-seed aggregation
comparison in Section 5. `mil_weighted_seed42` is a separate, independently
trained checkpoint with the same config/seed (trained by a different team
member) and is not the same set of weights.

| Model Metric (validation set) | Performance | Reference Baseline / Target |
| :--- | :---: | :---: |
| **Validation MAE (%)** | **2.6670** | Constant Mean MAE: 5.45% |
| **Validation RMSE (%)** | **3.5830** | Constant Mean RMSE: 6.85% |
| **Pearson Correlation ($r$)** | **0.8587** | - |
| **Spearman Rank Correlation ($\rho$)** | **0.8553** | - |
| **Pairwise Ranking Accuracy (Gap ≥ 5%)** | **0.9400 (94.0%)** | Pairwise severity ranking |

### Held-out test-set performance (previously missing from this report)

The 73-image closed test split was evaluated for the first time against the
current codebase on 2026-09-21 (`main.py --action evaluate_mil` against
`aggregation_weighted_seed42/checkpoints/best_model.pth`). There is a real,
previously unreported gap between validation and held-out test performance:

| Model Metric (test set, N=73) | `aggregation_weighted_seed42` | `mil_weighted_seed42` (independent re-train, same config) |
| :--- | :---: | :---: |
| **MAE (%)** | **2.5995** | 2.5902 |
| **RMSE (%)** | **3.7835** | 3.7782 |
| **Pearson Correlation ($r$)** | **0.7628** | 0.7546 |
| **Spearman Rank Correlation ($\rho$)** | **0.7653** | 0.7641 |
| **Pairwise Ranking Accuracy (Gap ≥ 5%)** | **0.9247 (92.5%)** | 0.9247 (92.5%) |

Both independently trained checkpoints agree closely with each other on the
test set (MAE within 0.01, Spearman within 0.001), which is a good
reproducibility signal. However, both show a materially lower Spearman
(~0.765) and Pearson (~0.76) on the true held-out test set than on validation
(~0.855 / ~0.859). MAE/RMSE hold up well (test is actually slightly better
than validation on MAE), but the held-out *rank correlation* is lower than
the validation number that has been used as the headline result so far.

A bootstrap check (5,000 resamples of the 73 test predictions,
`aggregation_weighted_seed42`) puts the test-set Spearman's 95% CI at
**[0.62, 0.86]** — the validation Spearman (0.855) falls at the upper edge of
this interval. With only 73 test samples, the val→test drop is plausibly
consistent with sampling noise rather than clear evidence of overfitting via
model/epoch selection, though it cannot rule overfitting out either. The
report should cite the test-set numbers (point estimate) as the primary
generalization claim, and should state the CI so the val→test gap reads as
an acknowledged small-sample uncertainty rather than an unexplained drop.

### Ranking-based training (`JointRankingRegressionLoss`), 3 seeds, 2026-09-27

Huber regression + pairwise `MarginRankingLoss` (margin 5, lambda 0.5). Backbone, area-weighted
plant-score pooling, head and all hyperparameters are identical to the §5 aggregation runs
(`configs/aggregation_experiments.json`); only `training_mode="joint"` differs. Retrained on the
cached 470-bag features as `outputs/runs/joint_weighted_seed{42,43,44}` (checkpoint metadata
records `training_mode: joint`). Both model families evaluated with the same script (val + test,
bootstrap = 5,000 resamples); the weighted val means reproduce §5 exactly.

| Model (3 seeds, mean ± SD) | Split | MAE (%) | Pearson r | Spearman rho | Pairwise acc. (gap >= 5) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Area-weighted, Huber (`aggregation_weighted_*`) | Val (N=66) | 2.6932 ± 0.0232 | 0.8558 ± 0.0026 | 0.8532 ± 0.0023 | 0.9397 ± 0.0009 |
| Area-weighted, joint ranking (`joint_weighted_*`) | Val (N=66) | 2.6589 ± 0.0424 | 0.8621 ± 0.0021 | 0.8591 ± 0.0005 | 0.9421 ± 0.0009 |
| Area-weighted, Huber | Test (N=73) | 2.6221 ± 0.0679 | 0.7502 ± 0.0194 | 0.7633 ± 0.0084 | 0.9242 ± 0.0060 |
| Area-weighted, joint ranking | Test (N=73) | 2.5145 ± 0.0533 | 0.7837 ± 0.0175 | 0.7810 ± 0.0020 | 0.9330 ± 0.0027 |

Per-seed test Spearman: Huber 0.765 / 0.771 / 0.754; joint 0.783 / 0.780 / 0.780 (bootstrap 95% CI
of each single run roughly [0.64, 0.88]).

Conclusion: the joint ranking loss is slightly but consistently better than pure regression on
every seed, on both validation and test. The gap is small relative to single-run CIs, so report
it as a consistent small improvement, not a large one. Protocol note: `aggregation_weighted_seed42`
was selected on validation *before* the test set was opened; the joint runs were trained afterwards,
but a validation-only choice between the two would also have picked the joint model.

Superseded: `patch_joint_seed42` / `patch_joint_sampled_seed42` (older on-the-fly patch pipeline,
single seed, `training_mode` not recorded) have **bit-identical weights** — one checkpoint was
overwritten. The sampled one scored test MAE 2.6051 / Spearman 0.7691. Do not cite them.

---

## 6b. Test-set results for all aggregation methods (2026-09-27)

From `scripts/evaluate_generalization.py` (dataset `GG1_calibration_test`; this path re-extracts
features from raw images and reproduces the cached-feature test metrics exactly). Mean ± SD, 3 seeds.

| Method | Test MAE | Test Spearman | Pairwise gap-5 |
| :--- | :---: | :---: | :---: |
| Uniform | **2.396 ± 0.013** | **0.808 ± 0.004** | 0.932 |
| ABMIL | 2.535 ± 0.036 | 0.770 ± 0.005 | 0.926 |
| Gated ABMIL | 2.548 ± 0.029 | 0.765 ± 0.007 | 0.924 |
| Area-weighted (selected on val) | 2.622 ± 0.068 | 0.763 ± 0.008 | 0.924 |
| Area-weighted + ranking loss | 2.515 ± 0.053 | 0.781 ± 0.002 | 0.933 |

On test, uniform pooling beats area weighting on every seed, reversing the validation ranking.
Reported as-is (no re-selection on test). Possible reason: raters likely average per-plant scores (Luca).

---

## 7. Zero-Shot Out-of-Distribution (OOD) Field Benchmark

**Naming correction (2026-09-27):** folder `2025_10_07_RSFB-Phenotyping_WG1_JLU` is **Weilburger
Grenze** (project brief data table), not Rauischholzhausen (`RHH1`, BBCH13, which has no scores
file). The label "Rauischholzhausen" in `evaluate_ood.py` and below is wrong.

**Extended study (2026-09-27, `scripts/evaluate_generalization.py`, outputs in
`outputs/tables/generalization/`):** all 16 checkpoints on WG1, DSV, and 218 held-out-plot GG1 images
where JLU/GAU disagree (>5 points). Reproduces the numbers below exactly for `mil_weighted_seed42`.
- Frame detected: GG test 100%, GG disagreement 99%, WG1 59%, DSV 0.1% (1/897). Plant regions per
  image: 6.0 / 7.3 / 14.3 / 24.4. Visual check: DSV uses another camera and the frame fills the
  photo; WG1 photos are taken from farther away with plants outside the frame.
- Area-weighted (3 seeds): WG1 MAE 4.32, rho 0.031 (constant median MAE 4.10 is better);
  DSV MAE 7.73, rho 0.264 (constant median 5.22 is better; mean prediction 11.8 vs true 7.1).
  No pooling variant exceeds rho 0.07 (WG1) / 0.27 (DSV). Plot-level rho -0.01 / 0.23.
- GG disagreement set: model rho 0.60 vs mean score; model vs JLU 0.50, vs GAU 0.55; JLU vs GAU
  only 0.27 (r 0.17). JLU mean 27.4% vs GAU 12.8% on these images.
- Brightness only matters at DSV (dark third MAE 10.2 vs bright third 6.5).

Zero-shot evaluation of `mil_weighted_seed42` on unseen field trial datasets. Re-run
2026-09-21 against the provenance-clean checkpoint (manifest SHA-256
`7048425afdb49fcd0fdf94c3c703b012bde008652e9ea2dcb08c7e7b9d3f24a8`, matching the
current frozen `baseline_manifest_split.csv`) after fixing an unrelated `create_splits`
dtype bug and a missing `argparse` import in `evaluate_ood.py`. This supersedes the
earlier preliminary numbers, which were flagged stale in `CONTINUATION_PLAN.md`.

| OOD Benchmark Dataset | Folder Identifier | Valid Samples ($N$) | MAE (%) | RMSE (%) | Pearson $r$ | Spearman $\rho$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Rauischholzhausen WG1** | `2025_10_07_RSFB-Phenotyping_WG1_JLU` | **906** | **4.33%** | **6.27%** | **0.113** | **0.032** |
| **DSV Trial 1** | `2025_09_15_Res4StRes_T1_DSV` | **897** | **8.11%** | **9.49%** | **0.309** | **0.270** |

Compared to the earlier preliminary run, DSV Trial 1 MAE improved substantially
(15.60% → 8.11%) and its Spearman correlation improved (0.193 → 0.270), while
Rauischholzhausen's already-weak rank correlation dropped further (0.199 → 0.032).
Overall conclusion is unchanged: the model generalizes poorly to unseen field trials,
and this confirmed (non-preliminary) result should be used in the report.

---

## 8. Classical Computer Vision Audits (Exploratory Work)

Audits evaluating classical computer vision algorithms as exploratory, explainable visual tools complementary to neural MIL regression.

| Classical CV Audit Module | Sample Size | Manual Usability Rate | Acceptance Status | Key Review Findings |
| :--- | :---: | :---: | :---: | :--- |
| **Plant Mask & Area Weight Audit** | 40 images | **100.0% (40/40)** | **Approved** | Plant masks accurately approximate visible leaf area for MIL weights. |
| **Direct Damage Audit** (Holes + Pitting) | 40 images | **0.0% (0/40)** | *Rejected for Holes* | Leaf masks & pitting are acceptable; automated shot-hole detection yields false positives. |
| **Soil-Aware Hole Detection Comparison** | 20 images | **0.0% (0/20)** | *Rejected* | Soil similarity thresholds confuse dark brown pitting with exposed soil below. |
| **RF-DETR Hole/Pitting Detector** | 40 images (32 train / 8 valid) | mAP@50 **3.47%** (was 0.39% before a labeling-pipeline bug fix, see below) | *Weak positive, not production-usable* | Predicted boxes now cluster on real leaf damage instead of soil/frame texture, but detector remains far too weak (~3-10% mAP@50) given only 283 training boxes on ~18px objects. |
| **Leaf/Cotyledon Count** (convexity-defect lobe counting) | 8 patches | not usable, not formally audited further | *Rejected* | Systematically overcounts a single damaged cotyledon as 2-3 "lobes" — a shot-hole/edge-bite notch is geometrically indistinguishable from a true inter-leaf notch, worst on exactly the simplest BBCH10-11 seedlings. Needs a learned segmentation model instead; see `analyses/PROJECT.md`. |

**Correction (2026-09-21):** the RF-DETR result above was originally 0.39% mAP@50 and attributed to "too little training data." That was largely wrong: `src/preprocessing/tile_hole_pitting_coco.py` read images with `PIL.Image.open()`, which auto-applies EXIF orientation, while the annotation coordinates are in the raw (un-rotated) pixel frame used everywhere else in this codebase — decorrelating essentially every box from its tile image. Fixing the image I/O to use `cv2` (matching the annotation frame) and retraining improved mAP@50 ~9x (0.39% → 3.47%), with qualitatively sane predictions. The two annotated classes are imbalanced (`shot_hole`: 222 train boxes, `pitting`: only 61); reaching a usable detector (mAP@50 ~30-50%) would likely need ~150-300 annotated images per class, 5-10x the current 40 — out of scope for this project's remaining time. See `analyses/HOLE_PITTING_ANNOTATION_PLAN.md` section 5 for the full investigation, four labeled/predicted example figures, and the data-requirement estimate.

---

## 9. Complete Project JSON Configuration Suite

The project configurations are organized into structured JSON files located in `configs/`:

### A. Core MIL Neural Pipeline Workflows
| Configuration File | Main Action (`action`) | Purpose & Description |
| :--- | :--- | :--- |
| **`configs/config_mil_data.json`** | `cache_embeddings` | Data preparation, split verification, and DINOv3 feature bag extraction (`outputs/cache/dinov3_bags.pt`). |
| **`configs/config_mil_train.json`** | `train_mil` | Area-Weighted MIL model training (`mil_weighted_seed42`), supporting `weighted`, `abmil`, and `gated_abmil`. |
| **`configs/config_mil_eval.json`** | `evaluate_ood` / `evaluate_mil` | In-distribution test set evaluation and zero-shot OOD field trial benchmarking. |
| **`configs/config.json`** | Master Default | Master fallback configuration used when `python main.py` is launched without parameters. |

### B. Baselines, Matrix Experiments & Pipeline Testing
| Configuration File | Main Action (`action`) | Purpose & Description |
| :--- | :--- | :--- |
| **`configs/config_mse.json`** | `train` | Whole-image baseline regression training using MSE loss. |
| **`configs/aggregation_experiments.json`** | `run_matrix` | 12-run grid experiment across 4 aggregation types (`weighted`, `uniform`, `abmil`, `gated_abmil`) and 3 seeds. |
| **`configs/smoke_config.json`** | Fast Smoke Test | Lightweight single-epoch configuration for rapid end-to-end pipeline verification. |

### C. Preprocessing & Classical Computer Vision Audits
| Configuration File | Audit Module | Purpose & Description |
| :--- | :--- | :--- |
| **`configs/frame_audit.json`** | Reference Frame Audit | Audit of metal reference frame cropping across 30 sample images. |
| **`configs/plant_region_audit.json`** | Plant Proposal Audit | Audit of plant region bounding box proposal across 30 sample images. |
| **`configs/plant_weight_audit.json`** | Area Weight Audit | Audit of plant mask pixel area weights across 40 sample images. |
| **`configs/direct_damage_audit.json`** | Direct Damage Audit | Audit of direct CV leaf area, hole, and pitting measurement across 40 images. |
| **`configs/hole_detection_comparison.json`** | Hole Classification Comparison | Comparison between HSV brightness vs CIELAB soil-aware hole detection across 20 difficult images. |
| **`configs/biology_audit.json`** | Biological Feature Audit | Audit of morphological biological feature extraction. |

---

## 10. Automated Verification & Reproducibility

- **Test Suite (`pytest`)**: **44 passed, 4 skipped in 28.96s**.
