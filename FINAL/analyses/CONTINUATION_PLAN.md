# CSFB Project Continuation Plan

## Purpose

This document defines the implementation order after completion of the official whole-image frozen-DINOv3 baseline. `analyses/PROJECT.md` remains the authoritative project-status document and must be updated after every implemented phase with results, commands, artifacts, limitations, and the next approved action.

## Working assumptions

- Analyze only plants inside the metal frame.
- Treat the 470 image scores as combined shot-hole and yellow/brown-pitting damage unless the supervisor clarifies otherwise.
- Preserve holes and pitting as separate concepts for later biological analysis.
- Weight plants according to visible plant/leaf area.
- Do not infer missing leaf-edge damage without dedicated manual validation.
- Develop using train and validation only; keep the 73-image test split closed until the complete new pipeline is selected and frozen.
- Never modify the read-only source dataset.

## Reference results

The new pipeline must be compared with the completed whole-image baseline:

- Best validation MAE: 5.1265
- Validation Spearman rho: 0.2712
- Test MAE: 4.6502
- Test RMSE: 6.2596
- Test Pearson r: -0.0971
- Test Spearman rho: -0.1078

The official test result is frozen and must not be used for iterative model selection.

## Phase 1 — Frame-interior extraction

Status: complete. The 30-image manual review was approved on 2026-08-30.

### Implementation

1. Detect the metal frame in each image.
2. Define the valid region inside the frame.
3. Exclude plants outside the frame.
4. Remove frame bars and the QR card where practical.
5. Save masks, crops, and overlays under `outputs/` without modifying source images.
6. Provide a deterministic command for regenerating the artifacts.

### Validation

- Review approximately 30 train/validation images.
- Include low, medium, and high damage scores.
- Include different lighting conditions and frame orientations.
- Record each result as successful, partial, or failed.
- Target at least 90% acceptable crops before continuing.

### Exit criteria

- The retained region matches the manually scored region.
- Outside plants and most QR/frame distractions are excluded.
- Failure cases are saved and handled by an explicit rule.
- Automated tests cover crop geometry and deterministic behavior.

Implementation artifacts:

- Configuration: `configs/frame_audit.json`
- Detector and audit generator: `src/preprocessing/frame_crop.py`
- Review table: `outputs/frame_audit/frame_audit.csv`
- Summary: `outputs/frame_audit/frame_audit_summary.json`
- Detection overview: `outputs/frame_audit/frame_audit_contact_sheet.png`
- Rectified-crop overview: `outputs/frame_audit/frame_crop_contact_sheet.png`

Final result: 30/30 detected and manually approved, with 15 train and 15 validation images and coverage of all four score ranges. The reviewer confirmed that the green polygons follow the outer metal square and that the rectified crops represent the intended inside-frame scoring region.

## Phase 2 — Plant-region detection

### Implementation

1. Detect green vegetation inside the frame using HSV or Lab color thresholds.
2. Clean masks with morphological operations.
3. Find connected plant or seedling regions.
4. Merge fragments that belong to the same nearby plant when necessary.
5. Reject implausible noise without removing small seedlings.
6. Generate padded, high-resolution plant crops.
7. Save masks, bounding boxes, overlays, and example crops.

Start with classical image processing. Introduce SAM, YOLO, or RF-DETR only if the simpler method is demonstrably inadequate.

### Validation

- Confirm that plants outside the frame are excluded.
- Confirm that small seedlings are retained.
- Confirm that holes and pitting remain visible in the crops.
- Inspect shadows, yellow plants, weeds, soil variation, and overlapping plants.
- Record empty-mask and over-segmentation failures.

### Exit criteria

- Plant proposals are usable across the reviewed sample.
- Every failure mode has a documented fallback.
- Source images remain unchanged.

## Phase 3 — Plant-patch dataset

Extend the data pipeline to return:

- image identifier and plot group;
- variable-length plant crops;
- visible area for each crop;
- image-level mean damage score;
- split identity and provenance.

Add tests for:

- images with no detected plants;
- one and multiple plant crops;
- variable crop counts within a batch;
- visible-area weighting;
- deterministic validation preprocessing;
- fixed grouped-split preservation;
- invalid or empty masks.

## Phase 4 — Plant-focused frozen-DINOv3 baseline

Use the following pipeline:

```text
Full image
  -> frame interior
  -> high-resolution plant patches
  -> frozen DINOv3 feature per patch
  -> patch aggregation
  -> regression head
  -> combined image damage percentage
```

Run these controlled train/validation comparisons:

1. Existing whole-image baseline.
2. Frame crop only.
3. Plant patches with uniform aggregation.
4. Plant patches with visible-area-weighted aggregation.

Keep the backbone, split, seed policy, checkpoint provenance, and evaluation protocol comparable.

### Required validation metrics

- MAE and RMSE.
- Pearson and Spearman correlations.
- Pairwise ranking accuracy at true-score gaps 5, 10, and 20.
- Prediction mean, standard deviation, minimum, and maximum.
- Error by target-score bin.
- Comparison with constant mean and median predictors.

### Selection requirements

The selected approach should:

- beat validation MAE 5.1265;
- improve clearly over validation Spearman 0.2712;
- beat constant predictors;
- reduce prediction-range compression;
- avoid relying on a single outlier or score bin;
- remain stable across multiple seeds.

## Phase 5 — Freeze the improved representation

Before continuing:

1. Select preprocessing and aggregation using validation results only.
2. Record all thresholds and fallback behavior in configuration.
3. Record the manifest hash, weight hash, environment, seed, and Git commit.
4. Freeze the primary model-selection rule.
5. Do not inspect or evaluate the test split.

If none of the plant-focused approaches improves validation performance, document the negative result before considering a more complex segmenter, higher-resolution backbone input, weak supervision, or domain adaptation.

## Phase 6 — Ranking-based experiment

Resume ranking only after the representation produces useful validation ordering.

1. Keep the selected plant-focused representation fixed.
2. Separate the pair-selection gap from the ranking-loss margin.
3. Exclude nearly equal scores.
4. Compare minimum pair gaps of 5, 10, and 20 percentage points.
5. Sample balanced pairs rather than materializing every possible pair.
6. Compare regression-only and joint regression-ranking objectives.
7. Run at least three seeds.
8. Report regression and ordering metrics together.

### Exit criteria

- Ranking improves validation ordering consistently across seeds.
- Any MAE tradeoff is reported explicitly.
- The final configuration is selected before test evaluation.

## Phase 7 — Final test evaluation

Once the complete new pipeline is frozen:

1. Record its configuration and Git commit.
2. Verify the data-manifest and DINOv3 weight hashes.
3. Use a new run name that does not overwrite the official baseline.
4. Perform one test evaluation.
5. Preserve and report the original whole-image baseline alongside the new result.
6. Save metrics, predictions, plots, logs, and provenance.

## Phase 8 — Additional biological features

After validating image-level scoring, develop and validate:

- number of plants inside the frame;
- visible area per plant;
- shot-hole count and area per plant;
- yellow/brown-pitting count and area per plant;
- combined affected area;
- pitting-to-hole ratio;
- approximate cotyledon/true-leaf count and BBCH stage.

Separate plant, leaf, hole, and pitting annotations will be required for quantitative validation. Do not infer separate severity weights from the combined 470 image-level labels.

## Immediate implementation order

1. Implement the frame-interior crop and overlay generator.
2. Generate a stratified 30-image train/validation review set.
3. Manually review and record crop quality.
4. Implement green-region masking and plant-patch generation.
5. Save and review patch overlays.
6. Add dataset and preprocessing tests.
7. Train the frame-crop and plant-patch validation experiments.
8. Compare all validation results with the official baseline and constants.
9. Decide whether the representation is ready for ranking.

## Documentation rule

After every implemented phase, update `analyses/PROJECT.md` with:

- phase status;
- code and configuration added;
- exact reproduction commands;
- artifact paths;
- validation sample and metrics;
- failures and limitations;
- manual decisions or supervisor clarifications;
- next approved step.

Do not mark a phase complete based only on code existing. Its exit criteria and proportional verification must also be complete.
