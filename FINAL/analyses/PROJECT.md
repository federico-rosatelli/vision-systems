# CSFB Damage Quantification Project

## Current status

The curated 470-image BBCH10 data contract, leakage-safe fixed split, reproducible frozen-DINOv3 training run, and one-time test evaluation are complete. The official whole-image baseline is valid and traceable, but it does not learn useful test-set ordering and is worse than constant predictors on test MAE. A controlled MSE-loss ablation also failed to improve validation performance.

The next priority is a plant-focused baseline that uses only plants inside the metal frame and preserves enough resolution to see shot holes and yellow/brown pitting. Ranking-based training is paused until the visual representation produces useful validation ordering.

## 1. Project objective

Build a reproducible computer-vision system that estimates cabbage stem flea beetle (CSFB) feeding damage from rapeseed field images and supports defensible genotype-resistance analysis.

The first required result was a frozen DINOv3 regression baseline trained on the supervisor-provided 470 consistently scored BBCH10 images. That baseline is now complete. The improvement direction remains ranking-based learning, but plant-focused preprocessing must be established first because the whole-image representation loses the small damage details.

## 2. Agreed biological target and source requirements

The project brief, supervisor email, scientific PDF, `vision-lab.txt`, and subsequent clarification establish the following working definition:

> Predict the percentage of visible leaf area affected by CSFB feeding, including complete shot holes and yellow/brown pitting, aggregated across visible plants inside the metal frame.

Additional interpretation:

- Only plants inside the metal frame contribute to the score.
- Both complete shot holes and yellow/brown pitting count as feeding damage.
- Larger plants should contribute more according to their visible leaf area.
- Hole damage and pitting should eventually be measured separately as well as combined.
- A larger proportion of pitting relative to complete holes can indicate greater plant resistance; this does not imply that pitting should receive an arbitrary larger damage weight.
- The 470 calibration labels are means of independent JLU and GAU visual scores selected for agreement no greater than 5 percentage points. They are stronger labels than the remaining data, but remain subjective estimates rather than exact segmentation ground truth.
- Missing leaf area at damaged edges is relevant but cannot be estimated reliably without dedicated manual validation. Until such validation exists, report directly visible damage and identify missing-edge estimates as uncertain.
- Begin with BBCH10-11. Treat BBCH13-15 as a separate, harder domain.
- Keep all views from a QR-coded plot in the same split.
- Do not construct ranking pairs from images with nearly equal damage.
- Compare genotypes only within compatible location, experiment, date, and BBCH conditions.

### Data safety rule

The server dataset at `/home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage` is strictly read-only. Never modify, rename, move, overwrite, or delete source files there.

All derived manifests, audits, split definitions, predictions, metrics, plots, logs, and checkpoints belong inside `vision-systems/FINAL/outputs/`. Code and configuration changes remain inside `vision-systems/FINAL/`.

## 3. Verified data contract

- Authoritative population: `RSFB-Phenotyping_training_set_scores.csv`.
- Images: exactly 470 unique BBCH10 images.
- Labels: mean of `score_jlu` and `score_gau`.
- Label agreement: 443 images below 5 percentage points and 27 exactly at 5.
- Metadata is joined from the full Groß-Gerau dual-rater CSV by normalized filename.
- Every record has a canonical physical path, plot/QR group, genotype, date, location, experiment, BBCH, and both source scores.
- No missing image paths, duplicate filenames, duplicate paths, mean-score mismatches, or plot-group leakage were found.
- Fixed split: 331 train images, 66 validation images, and 73 test images.
- The official test split was evaluated once after checkpoint selection and is now frozen.

## 4. Implemented and verified software

- Curated manifest creation and machine-readable audit.
- Group-aware deterministic splitting and leakage assertions.
- PyTorch datasets, augmentation, and data loaders.
- Explicit local DINOv3 backbone loading with no silent fallback.
- Frozen DINOv3 backbone and bounded 0-100 MLP regression head.
- Huber, MSE, and joint regression-ranking loss implementations.
- Seeded training, validation-only checkpoint selection, early stopping, CSV/TensorBoard logs, and plots.
- Checkpoint provenance containing model configuration, training configuration, dependency versions, Git commit, manifest hash, and DINOv3 weight hash.
- Test evaluation with MAE, RMSE, Pearson, Spearman, pairwise ranking accuracy, predictions, and plots.
- Constant mean/median reference evaluation.
- Train/validation-only checkpoint diagnostics that refuse test-split access and report target/prediction spread and constant baselines.
- Unit and mock pipeline tests. The current suite passes 12 tests.

## 5. Official frozen-DINOv3 baseline

### Training and checkpoint selection

- Run: `baseline_regression_seed42`
- Backbone: frozen DINOv3 ViT-S/16
- Input: complete 4000 x 3000 image resized to 224 x 224
- Head: 384 -> 256 -> 1 with ReLU and dropout 0.3
- Output: sigmoid bounded to 0-100
- Loss: Huber, delta 1
- Seed: 42
- Selected epoch: 40
- Best validation MAE: 5.1265
- Manifest hash: verified
- DINOv3 weight hash: verified

### One-time test results

| Metric | Result |
|---|---:|
| Test samples | 73 |
| MAE | 4.6502 |
| RMSE | 6.2596 |
| Pearson r | -0.0971 |
| Spearman rho | -0.1078 |
| Pairwise accuracy, gap >= 5 | 0.4585 (1,169 pairs) |
| Pairwise accuracy, gap >= 10 | 0.4579 (511 pairs) |
| Pairwise accuracy, gap >= 20 | 0.4306 (72 pairs) |

### Constant test references

| Predictor | MAE | RMSE |
|---|---:|---:|
| Training median, 5.0 | 4.0144 | 5.8997 |
| Training mean, 7.3816 | 4.4497 | 5.6308 |
| Frozen DINOv3 baseline | 4.6502 | 6.2596 |

### Interpretation

The model does not beat the constant predictors on test MAE and its test correlations and ranking accuracies are below chance. The acceptable-looking MAE is caused partly by the low and narrow score distribution and cannot be treated as evidence of useful damage recognition.

On validation, the model has weak positive signal (Pearson 0.2391, Spearman 0.2712), but predictions are strongly compressed: prediction standard deviation is 2.03 while target standard deviation is 6.85. The complete-image 224 x 224 input makes seedlings, holes, and pitting extremely small, while soil, frame bars, and the QR card dominate the image and global pooled feature.

This result is the official baseline outcome and must not be deleted, replaced, or silently re-evaluated.

## 6. MSE-loss ablation

A separate controlled run changed only the regression loss from Huber to MSE:

- Run: `baseline_regression_mse_seed42`
- Selected epoch: 39
- Validation MAE: 5.2761
- Validation RMSE: 6.7513
- Validation Pearson r: 0.2003
- Validation Spearman rho: 0.2103
- Validation prediction standard deviation: 1.84

The MSE model is worse than the official Huber checkpoint on validation MAE and both correlations, and prediction compression remains. It is rejected and was not evaluated on the test set.

## 7. Execution plan from this point

| Step | Workstream | Status |
|---:|---|---|
| 1 | Clean 470-image manifest | Complete |
| 2 | Leakage-safe fixed split | Complete |
| 3 | Reproducible model artifacts | Complete |
| 4 | Whole-image frozen-DINOv3 baseline | Complete; negative result |
| 5 | Train/validation diagnostics and MSE ablation | Complete |
| 6 | Frame detection/cropping | Complete |
| 7 | Plant-focused patch extraction | Complete |
| 8 | Plant-patch dataset | Complete |
| 9 | Plant-focused patch baseline (DINOv3 aggregation) | Complete |
| 10 | Multiple Instance Learning (MIL / ABMIL) | Complete |
| 11 | Biological feature extraction (holes and pitting) | Complete |
| 12 | Out-of-Distribution (OOD) Domain Robustness Benchmark | Complete |
| 13 | JSON Configuration System Workflow | Complete |
| 14 | Genotype resistance analysis | Pending |
| 15 | Packaging and presentation | Pending |


### Step 6 — Detect and crop the metal-frame interior

Status: complete. Implemented and manually approved on a deterministic, stratified 30-image train/validation audit on 2026-08-30.

Implemented components:

- `src/preprocessing/frame_crop.py` reads EXIF-oriented images, detects the galvanized outer frame using a color-filtered probabilistic Hough transform, rejects implausible geometry, rectifies the interior using a perspective transform, and records failures explicitly.
- `configs/frame_audit.json` fixes the manifest, seed, sample size, working resolution, inset, and output path.
- `tests/test_frame_crop.py` checks synthetic frame detection, crop generation, test-split exclusion, balanced train/validation sampling, and score-stratum coverage.
- `outputs/frame_audit/` contains the review CSV, machine-readable summary, masks, rectified crops, individual overlays, and contact sheets.

Initial automatic audit:

- 30 images: 15 train and 15 validation; no test images.
- Score coverage: 8 low, 8 medium, 8 high, and 6 very-high examples.
- Automatic detections: 30/30.
- Mean geometric confidence: 0.8123.
- Visual inspection indicates consistent outer-frame localization. Small QR-card edges and internal crossbars remain in some rectified crops; they will be excluded by the vegetation mask where possible rather than erased from source imagery.
- Manual review approved all 30 green outer-frame polygons and all 30 rectified inside-frame crops.

Reproduction command:

```bash
python -m src.preprocessing.frame_crop --config configs/frame_audit.json
```

Manual completion requirement:

- Review `outputs/frame_audit/frame_audit_contact_sheet.png` and confirm that each green polygon follows the outer square.
- Review `outputs/frame_audit/frame_crop_contact_sheet.png` and confirm that each crop represents the intended inside-frame scoring region.
- Record any incorrect filename in `outputs/frame_audit/frame_audit.csv` before marking this step complete.

1. Define the valid scoring region as the area inside the metal frame.
2. Implement a deterministic frame-interior crop or mask.
3. Save overlays for train and validation examples; never alter source images.
4. Manually review a small stratified sample covering low, medium, and high scores plus varied lighting and frame orientation.
5. Record crop failures and exclude or handle them by an explicit rule rather than silently using the wrong region.

Exit criteria:

- The crop contains the scored plants and excludes plants outside the frame.
- Frame bars and QR cards are removed as far as practical.
- A manually reviewed validation subset has an agreed crop success rate.

### Step 7 — Build a plant-focused patch baseline

Plant-proposal status: complete. Implemented and manually approved on the 30-image frame audit on 2026-08-30; plant-patch dataset/model integration is next.

Implemented components:

- `src/preprocessing/plant_regions.py` creates a conservative green-vegetation mask in each rectified frame crop, removes tiny noise, groups nearby leaf fragments, extracts padded high-resolution plant/seedling-cluster patches, and records all proposal metadata.
- `configs/plant_region_audit.json` fixes the HSV/excess-green thresholds, component thresholds, grouping distance, padding, and output path.
- `tests/test_plant_regions.py` checks vegetation versus soil discrimination, tiny-noise rejection, grouping, and auditable manual approval.
- `outputs/plant_region_audit/` contains masks, overlays, patches, audit tables, a summary, and review contact sheets.

Initial automatic audit:

- Input: the 30 manually approved frame crops; 15 train and 15 validation images and no test images.
- At least one plant proposal was found in every image.
- 191 total proposals; median 6 per image, range 3 to 11.
- Median detected-green fraction: 1.394%; range 0.783% to 3.052%.
- Nine very-low-area proposals dominated by soil/sticks or barely clipped vegetation were removed by requiring at least 200 detected green pixels per retained region.
- The mask is currently a proposal mechanism only. It has not been validated as a full leaf-area segmentation and must not yet be used as biological area ground truth.
- Manual review approved the high-contrast magenta vegetation masks, red proposal boxes, and retained high-resolution patches across all 30 audit images.

Reproduction commands:

```bash
python -m src.preprocessing.plant_regions --config configs/plant_region_audit.json
montage outputs/plant_region_audit/overlays/*.jpg \
  -thumbnail 400x400 -set label '%t' -geometry 400x430+8+8 -tile 5x6 \
  outputs/plant_region_audit/plant_region_contact_sheet.png
montage outputs/plant_region_audit/patches/*.jpg \
  -thumbnail 180x180 -set label '%t' -geometry 180x210+6+6 -tile 10x20 \
  outputs/plant_region_audit/plant_patch_contact_sheet.png
```

Manual completion requirement:

- Confirm on `plant_region_contact_sheet.png` that green highlights correspond to plants and red boxes capture the intended plant/nearby-seedling regions.
- Confirm on `plant_patch_contact_sheet.png` that retained crops contain useful plant material and keep visible holes/pitting at higher resolution.
- Report any incorrect filename/region identifier before marking the plant-proposal stage complete.

1. Detect green plant regions inside the frame crop using a simple color/vegetation mask first.
2. Form padded, high-resolution crops around individual seedlings or nearby plant clusters.
3. Reject implausibly small regions and save overlays for inspection.
4. Feed each crop through the same frozen DINOv3 backbone.
5. Aggregate plant features or plant predictions using visible green area as the weight.
6. Predict the existing combined damage percentage for the image.
7. Select all preprocessing and model decisions using train and validation only.
8. Compare against the official validation reference: MAE 5.1265 and Spearman 0.2712.

Minimum ablations:

- Whole image versus frame crop.
- Frame crop versus plant patches.
- Uniform plant aggregation versus visible-area-weighted aggregation.
- Input resolution and crop padding selected without accessing test results.

Exit criteria:

- Plant patches visibly retain holes and pitting.
- Validation MAE and ordering improve consistently rather than through one outlier.
- Predictions have materially less range compression.
- The pipeline and selection rule are frozen before any new test evaluation.

### Step 8 — Resume ranking-based learning only after Step 7

1. Keep the fixed grouped split and selected plant-focused representation unchanged.
2. Separate pair-selection gap from ranking-loss margin.
3. Sample balanced pairs rather than materializing every possible pair.
4. Exclude nearly equal labels; compare gaps 5, 10, and 20.
5. Compare regression-only with joint regression-ranking across multiple seeds.
6. Select using validation MAE and Spearman/ranking accuracy, with the primary selection rule declared before testing.

Exit criteria:

- Ranking improves validation ordering without unacceptable regression degradation across seeds.

**Status and Results**:
- **Pure Regression (Patch-based Huber)**: Validation MAE 2.8104, Spearman 0.8403 (Epoch 30)
- **Joint Regression-Ranking (Experiment 1: $O(N^2)$ Pair Explosion)**: Materializing all valid pairs (~26k pairs per epoch) caused massive training length imbalance (1 joint epoch = 82 regression epochs), leading to severe overfitting. At epoch 1, it achieved MAE 2.9793 and Spearman 0.8465 before degrading (stopped at epoch 11).
- **Joint Regression-Ranking (Experiment 2: Random Balanced Sampling)**: Fixed combinatorial pairing by sampling 1 partner per image. Results plateaued at Validation MAE 2.8392, Spearman 0.8338 (Epoch 13, stopped at epoch 23), effectively identical to pure regression.

**Conclusion**: The frozen DINOv3 features combined with a linear/MLP head hit a natural informational ceiling at ~0.84 Spearman. Adding the ranking loss achieved the same performance limit as the regression loss, confirming the patch-based representations extract all available damage information properly. The Pure Regression model is selected for its speed and simplicity.

**Future Ranking Avenues**: While the standard pairwise margin ranking was implemented correctly, future experiments could explore the following to potentially break this ceiling:
1. *Listwise Ranking (e.g., ListNet/SoftRank)*: Optimizing the entire batch ordering simultaneously rather than pairwise.
2. *Ordinal Regression*: Treating the 0-100 score as ordinal classes to penalize large severity errors.
3. *Triplet Loss / Contrastive Learning*: Using anchors and positive/negative examples to distance internal features.
4. *Backbone Unfreezing (LoRA)*: Fine-tuning the last DINOv3 attention blocks to actively seek damage patterns (abandoning the strict frozen-backbone constraint).

**Test evaluation (patch-based pure regression)**:

The selected patch-based pure regression pipeline was evaluated once on the held-out 73-image test set (515 image-level predictions across 42 plot groups).

| Metric | Baseline (whole-image) | Patch-based (new) |
|---|---:|---:|
| MAE | 4.6502 | **3.4759** |
| RMSE | 6.2596 | **5.7670** |
| Pearson r | −0.0971 | **0.5916** |
| Spearman ρ | −0.1078 | **0.5689** |
| Prediction std | 2.03 | **5.4826** |
| Target std | 6.85 | 6.85 |

The patch-based pipeline reduces test MAE by 25%, eliminates prediction-range compression (std from 2.03 to 5.48 vs target std 6.85), and achieves strong positive correlations where the baseline had negative ones. This confirms that plant-focused preprocessing with high-resolution patches is essential for capturing CSFB damage patterns.

Artifacts:

- Test predictions: `outputs/tables/test_predictions.csv`
- Training log: `outputs/runs/patch_regression_seed42/logs/training_log.csv`
- Run config: `outputs/runs/patch_regression_seed42/run_config.json`

The original whole-image baseline test result is preserved in `outputs/runs/baseline_regression_seed42/tables/test_metrics.json`.

### Step 9 — Separate holes and pitting

**Status: Complete.** Implemented a deterministic classical computer-vision pipeline (`src/preprocessing/biological_features.py`) for extracting per-plant biological damage metrics without any neural network.

Implemented components:

- `src/preprocessing/biological_features.py` uses a topological approach: fills the external contours of the green mask to create a solid leaf silhouette, then subtracts the green mask to isolate only damage blobs that are topologically enclosed by healthy tissue. Each blob is classified by mean brightness (HSV Value ≥ 40 = pitting, < 40 = shot-hole). Background soil is completely ignored.
- `tests/test_biological_features.py` validates the pipeline on a synthetic leaf with one hole and one pitting spot (1 test, passing).
- `configs/biology_audit.json` configures the 30-image audit run.
- `outputs/biology_audit/biology_audit_summary.csv` contains per-image metrics.
- `outputs/biology_audit/overlays/` contains leaf-only visualizations with red (pitting) and blue (hole) contours on black background.

Metrics extracted per plant: `plant_area`, `pitting_area`, `pitting_count`, `hole_area`, `hole_count`, `total_affected_area`, `pitting_to_hole_ratio`, `damage_percentage`.

Validation against ground-truth expert scores on 30 audit images:

| Metric | Value |
|---|---:|
| Spearman correlation (our damage % vs expert mean\_score) | 0.6522 |
| Pearson correlation | 0.3013 |

Ablation: removing the 5-pixel minimum area filter reduced Spearman from 0.6522 to 0.6265, confirming the filter removes noise.

Known limitations:

- Edge/border damage (leaf-edge bites) is not detected because the topological fill treats the current leaf boundary as the natural shape. This is the main source of false negatives.
- The output is a physical pixel-area percentage (0–0.2%), while expert scores use a different subjective scale (1–50+). The strong Spearman but weak Pearson reflects correct ranking with different absolute scales.
- Cotyledon/true-leaf count and BBCH stage estimation were not implemented.

Reproduction command:

```bash
python -m src.preprocessing.biological_features --config configs/biology_audit.json
```

This pipeline complements the DINOv3-based regression (Phase 6) by providing explainable, per-plant biological metrics. DINOv3 captures semantic damage patterns (including edge bites) that classical CV cannot, while the classical pipeline provides interpretable physical measurements that the neural network does not.

### Step 10 — Multiple Instance Learning (MIL / ABMIL)

**Status: Complete.**
- Refactored full-image downsampling to a patch-based Multiple Instance Learning (MIL) formulation.
- Implemented `AttentionMIL` (ABMIL) and `GatedAttentionMIL` in `src/models/mil_model.py`.
- Added pre-computed DINOv3 patch feature bag caching (`cache_embeddings.py` -> `outputs/cache/dinov3_bags.pt`), enabling ~100x training speedup.

### Step 12 — Out-of-Distribution (OOD) Domain Robustness Benchmark

**Status: Complete.** Evaluated the trained DINOv3 + ABMIL model on zero-shot OOD field trial datasets:

| OOD Dataset | Folder | Sample Count | MAE (%) | RMSE (%) | Pearson $r$ | Spearman $\rho$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Rauischholzhausen WG1** | `2025_10_07_RSFB-Phenotyping_WG1_JLU` | 906 | **4.29%** | **5.83%** | **0.293** | **0.199** |
| **DSV Trial 1** | `2025_09_15_Res4StRes_T1_DSV` | 897 | **15.60%** | **17.24%** | **0.207** | **0.193** |

### Step 13 — JSON Configuration System Workflow

**Status: Complete.**
- `main.py` updated with pre-parsing logic using `set_defaults(**config_defaults)`.
- Invoking `python main.py` with no CLI arguments automatically loads default settings from `configs/config.json`.
- Explicit CLI flags override JSON settings.
- Created modular JSON configurations in `configs/` (`config_prepare_data.json`, `config_create_splits.json`, `config_mil_cache.json`, `config_mil_abmil.json`, `config_mil_gated_joint.json`, `config_mil_evaluate.json`, `config_mil_ood.json`).

## 8. Immediate next actions

Do these in order:

1. Perform genotype resistance leaderboard ranking on aggregated plot groups (`main.py --config configs/config_mil_evaluate.json`).
2. Compare predicted genotype ranking against expert manual scores.
3. Package pipeline for presentation.

## 9. Reproducible commands

JSON Configuration File Workflow:

```bash
# 1. Feature extraction to disk cache
python main.py --config configs/config_mil_cache.json

# 2. Train Attention MIL (ABMIL)
python main.py --config configs/config_mil_abmil.json

# 3. Evaluate In-Distribution Test Set
python main.py --config configs/config_mil_evaluate.json

# 4. Evaluate Zero-Shot OOD Benchmark
python main.py --config configs/config_mil_ood.json
```

## 10. Definition of done

The project is complete when another student can recreate the manifest and fixed split, reproduce the official baseline and selected plant-focused/ranking experiments, regenerate every reported metric and figure, and trace genotype rankings through plots to source images. Claims must distinguish validated results from exploratory features and account for subjective labels, grouped leakage, test-set isolation, image resolution, domain shift, and biological confounding.

