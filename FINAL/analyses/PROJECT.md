# CSFB Damage Quantification Project

## Current status

The software prototype is substantially implemented, but the official experiment is not yet valid or reproducible. The immediate priority is correcting the 470-image baseline manifest and grouped split. Reported model metrics and genotype results remain unverified until their real artifacts are reproduced.

## 1. Project objective

Build a reproducible computer-vision system that estimates cabbage stem flea beetle damage from rapeseed field images and supports defensible genotype-resistance analysis.

The required first result is a frozen DINOv3 regression baseline trained on the supervisor-provided 470 high-quality BBCH10 images. The selected improvement direction is ranking-based learning. Plant-, leaf-, hole-, and pitting-level features are later extensions.

## 2. Source requirements

The project brief, supervisor email, supplied scientific PDF, and `vision-lab.txt` establish these requirements:

- Begin with damage scoring on younger plants, especially BBCH10-11.
- Use DINOv3 as the frozen baseline backbone.
- Use the supervisor-curated 470-image training set. Its file contains 443 images with disagreement below 5 percentage points and 27 with disagreement exactly 5.
- Keep the three views of each QR-coded plot in the same data split.
- Do not construct ranking pairs from images with nearly equal damage.
- Treat BBCH13-15 as a harder, separate domain.
- Aggregate plant estimates by visible area when plant-level scoring is introduced.
- Distinguish shot holes from yellow/brown pitting where possible.
- Do not claim reliable missing-edge damage estimates without manual validation.
- Compare genotypes only under compatible location, experiment, date, and BBCH conditions.

### Data safety rule

The server dataset at `/home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage` is strictly read-only. Never modify, rename, move, overwrite, or delete any source file or directory there.

All derived manifests, audits, split definitions, predictions, metrics, plots, logs, and model checkpoints must be written inside `vision-systems/FINAL/outputs/`. Project code and configuration changes must remain inside `vision-systems/FINAL/`.

## 3. Work already implemented

### Verified implementation

- Repository structure for data, models, training, evaluation, inference, visualization, configuration, and tests.
- CSV parsing and manifest-generation code.
- Group-aware train/validation/test splitting code.
- PyTorch image dataset, augmentation, and data loaders.
- Frozen vision-backbone model with an MLP regression head and bounded 0-100 output.
- Huber/MSE regression and joint regression-ranking loss implementations.
- Training loop with seeding, checkpointing, early stopping, CSV/TensorBoard logging, and plots.
- Evaluation code for MAE, RMSE, Pearson correlation, Spearman correlation, predictions, and residual plots.
- Single-image inference command.
- Initial plot/genotype aggregation code.
- Unit and mock pipeline tests.
- A committed model checkpoint and generated manifest files.

### Available commands

The pipeline is controlled through `main.py`:

```bash
python main.py prepare_data --config configs/config.json
python main.py train --config configs/config.json
python main.py evaluate --config configs/config.json
python main.py rank --config configs/config.json
python main.py predict --image IMAGE.jpg --config configs/config.json
python main.py plot_logs --config configs/config.json
python main.py test --config configs/config.json
```

These commands describe the implemented interface. They should not be treated as a reproducible final workflow until Steps 1-3 below are completed.

### Existing but not yet verified as final results

- A reported MAE of 3.47% and Spearman correlation of 0.57.
- Claims that regression and joint-ranking experiments were compared.
- Claims that a final genotype leaderboard was produced.

These claims currently lack committed real prediction tables, metrics, logs, experiment configurations, plots, and leaderboard artifacts.

## 4. Current data findings

- The real dataset is located at `/home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage`.
- It contains 18,457 JPG files in the local nested copy, compared with 8,946 images documented in the brief; duplicate/nested image copies must be reconciled.
- Four score CSVs exist: two single-rater field datasets, one 936-row Groß-Gerau dual-rater dataset, and one curated 470-row training-set CSV.
- The authoritative baseline list is `RSFB-Phenotyping_training_set_scores.csv`, containing exactly 470 data rows.
- Its QR code and genotype metadata should be joined from `2025_10_21_RSFB-Phenotyping_GG1_scores.csv` by normalized filename.
- The current manifest recursively merges both CSVs, causing duplicate baseline records.
- The current committed manifest contains 886 usable high-quality rows rather than the required 470.
- A shared missing value, `unknown`, groups 443 test images together and produces an invalid 305/66/515 train/validation/test image distribution.
- All committed BBCH values are `Unknown`, and nearly all extracted dates are missing.
- The manifest uses `Genotyp`, while the ranking code expects `genotype`.

## 5. Execution plan

| Step | Workstream | Status |
|---:|---|---|
| 1 | Clean 470-image baseline manifest | Complete |
| 2 | Leakage-safe fixed split | Complete |
| 3 | Reproducible model artifacts | Complete |
| 4 | Official DINOv3 regression baseline | Next |
| 5 | Ranking-based comparison | Prototype exists; experiment pending |
| 6 | Domain robustness | Pending |
| 7 | Genotype resistance analysis | Prototype exists; validation pending |
| 8 | Agronomic feature extraction | Optional stretch |
| 9 | Packaging and presentation | Pending |

### Step 1 — Rebuild the baseline data contract

Status: complete.

1. Read the curated 470-row CSV as the only baseline population.
2. Normalize filename extensions and case.
3. Join QR code, genotype, plot, row, and column metadata from the full Groß-Gerau dual-rater CSV.
4. Resolve each image to one canonical physical path rather than mapping duplicate basenames arbitrarily.
5. Add fixed metadata: location, experiment, sampling date, and BBCH10.
6. Reject or explicitly audit missing scores, paths, QR codes, genotypes, and duplicate image identities.
7. Save a baseline manifest plus a machine-readable audit report.

Exit criteria:

- Exactly 470 unique manifest rows.
- Every row has two scores, mean score, disagreement no greater than 5, canonical image path, QR/plot group, genotype, date, location, and BBCH.
- No duplicate image identities or unresolved shared `unknown` group.

### Step 2 — Create a valid fixed split

Status: complete.

1. Split by QR/plot group, never by image.
2. Target approximately 70/15/15 train/validation/test proportions while balancing image counts and score bins.
3. Save split group IDs separately for reproducibility.
4. Add assertions for group leakage, duplicate leakage, missing groups, class balance, and expected population size.

Exit criteria:

- No plot or duplicate image crosses splits.
- Split proportions and score distributions are documented and reasonable.
- The same split is reused by every experiment.

### Step 3 — Make model artifacts reproducible

Status: complete.

1. Require an explicit DINOv3 backbone; remove silent DINOv2 fallback from official runs.
2. Store model name, preprocessing, image size, head architecture, training mode, loss settings, seed, data hash, and Git commit in every checkpoint.
3. Use separate output directories for every experiment and seed.
4. Add environment setup and exact commands to a README or run guide.

Exit criteria:

- A checkpoint can be loaded without guessing its architecture or training configuration.
- Every reported number is traceable to one run directory.

### Step 4 — Run the official regression baseline

Status: waiting for authorized DINOv3 weights and GPU access. Mean/median references are complete.

1. Calculate training-set mean and median predictor metrics.
2. Train the frozen DINOv3 regression head on the corrected training split.
3. Select the checkpoint using validation MAE only.
4. Evaluate once on the untouched test split.
5. Save metrics, predictions, learning curves, predicted-versus-true plots, residuals, score-bin errors, and best/worst examples.
6. Compare model error with inter-rater disagreement.

Required metrics:

- MAE and RMSE.
- Pearson and Spearman correlation.
- Pairwise ranking accuracy at multiple true-score gaps.
- Sample counts and confidence intervals.

Exit criteria:

- A reproducible DINOv3 baseline report suitable for sending to the supervisor.

### Step 5 — Evaluate ranking-based learning

1. Separate the pair-selection gap from the ranking-loss margin.
2. Sample balanced pairs rather than materializing every possible pair.
3. Compare gaps such as 5, 10, and 20 percentage points.
4. Compare regression-only, ranking-only, and joint regression-ranking objectives.
5. Keep the split, backbone, preprocessing, and evaluation protocol identical.
6. Run multiple seeds and report uncertainty.

Exit criteria:

- A controlled ablation table establishes whether ranking improves regression or ordering performance consistently.

### Step 6 — Test domain robustness

1. Evaluate the selected model separately on other labeled BBCH10-11 locations.
2. Evaluate BBCH13-15 separately as a domain-shift test.
3. Report results by location, date, lighting condition where available, and BBCH.
4. Do not combine these results into the original held-out baseline score.

Exit criteria:

- Clear evidence of where the model transfers and where it fails.

### Step 7 — Produce defensible genotype rankings

1. Aggregate the three image views into plot-level predictions.
2. Retain view-to-view variation as an uncertainty measure.
3. Compare genotypes within compatible location/date/BBCH/experiment blocks.
4. Aggregate replicated plots and report mean adjusted damage, confidence interval, plot count, and rank.
5. Compare predicted rankings against manual-score rankings using Spearman correlation, pairwise accuracy, and top/bottom overlap.

Exit criteria:

- A traceable leaderboard with uncertainty and adequate replication, not merely a sorted list of test images or plots.

### Step 8 — Add agronomic features

Status: stretch work after the scoring experiments are valid.

1. Segment plants and the metal-frame region.
2. Count plants inside the frame.
3. Estimate plant area and use area-weighted aggregation.
4. Prototype shot-hole and yellow/brown-pitting masks or detectors.
5. Estimate leaf counts or BBCH class.
6. Validate every feature against a small manually annotated test set.

Exit criteria:

- Quantitative validation and overlays distinguish reliable features from exploratory outputs.

### Step 9 — Package and present

1. Provide image and folder inference with CSV export.
2. Optionally add a small Streamlit or Gradio interface.
3. Freeze final configs, split IDs, checkpoints, metrics, and figures.
4. Write a concise report covering data quality, methods, experiments, failures, limitations, and biological interpretation.
5. Prepare a 10-12 slide presentation and recorded demo backup.

## 6. Immediate next actions

Do these now, in order:

1. Verify DINOv3 access and run a short one-epoch smoke test.
2. Calculate mean and median reference metrics.
3. Run the official regression training in `outputs/runs/baseline_regression_seed42/`.
4. Evaluate the selected checkpoint once on the untouched test split.
5. Save and review all metrics, predictions, plots, and failure examples.

Do not begin UI or segmentation work until the corrected baseline experiment is complete.

## 7. Definition of done

The project is complete when another student can recreate the manifest and fixed split, train and evaluate the baseline and best ranking extension, reproduce every reported metric and figure, and trace genotype rankings back through plots to source images. Claims must clearly distinguish validated results from exploratory features and must account for label disagreement, group leakage, domain shift, and biological confounding.
