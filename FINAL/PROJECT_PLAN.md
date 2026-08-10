# CSFB Feeding-Damage Quantification: Project Plan

## 1. Goal and Scope

Build a reproducible vision pipeline that predicts **average cabbage stem flea beetle (CSFB) damage (%)** from field images of young rapeseed and uses those predictions to identify **which cultivars/genotypes are more resistant under comparable field conditions**. First deliver the required DINOv3 supervised baseline; then improve weak-label use with ranking-based learning and add interpretable plant/damage features.

**Required final outputs:** trained baseline and improved model, held-out results, error analysis/visualizations, inference script or small analysis UI, reproducible repository, short report, and presentation/demo.

**Target outputs:**

- Primary: image-level damage score in `[0, 100]`, with visual evidence highlighting suspected holes and yellow/brown pitting.
- Biological decision output: plot-level and genotype-level resistance ranking, where lower adjusted damage means greater resistance.
- Extended: plants inside frame; leaves per plant; shot-hole and yellow/brown-spot count and area; area-weighted damage per plant/image.
- Initial scope: BBCH10-11. Treat BBCH13-15 as a later, separate domain because of larger overlapping leaves and edge damage.

## 2. Technical Decisions

- **Baseline:** frozen pretrained DINOv3 image encoder plus a small regression head, trained on the 470 high-quality images. Predict a bounded score (for example `100 * sigmoid(logit)`) with Huber or MSE loss.
- **Primary extension:** ranking-based training. Generate pairs only when label difference exceeds a validated margin (start at 5 percentage points); optimize pairwise BCE/Bradley-Terry loss, optionally together with regression loss.
- **Aggregation unit:** keep image predictions, then aggregate the three views belonging to one QR-coded plot. Never split views of one plot across train/validation/test.
- **Resistance comparison:** first average the three images per plot, then compare genotypes only within compatible location, experiment, date, and BBCH groups. Report rank and uncertainty; do not interpret raw cross-location differences as genetic resistance.
- **Primary model-selection metric:** validation MAE. Also report RMSE, Pearson `r`, Spearman rank correlation, and pairwise ranking accuracy. Compare model MAE with human disagreement.
- **Interpretability:** show prediction/label pairs, residual plots, score-bin performance, and attention/nearest-neighbor examples. Do not claim that attention is a damage segmentation.

## 3. Phased Implementation

### Phase 0 - Reproduce the Data Contract

1. Inventory the dataset at `/home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage`; reconcile the documented 8,946 images with local duplicate/nested folders before training.
2. Parse all three score CSVs robustly (`;` versus `,`, BOM, missing `.jpg` suffix). Create one manifest with `path`, filename, QR/plot, location, date, BBCH, rater scores, mean score, disagreement, and `is_high_quality`.
3. Define the 470 calibration samples as `diff < 5` (verify exact count and image joins). Use `(Score_JLU + Score_GAU) / 2` as the target; retain each rater score for uncertainty analysis.
4. Validate every path, score range, duplicate hash, QR group size, image dimensions, corrupt file, and class/score distribution. Save an audit table and representative image grid.

**Exit criterion:** deterministic manifest and audit report; all exclusions have logged reasons.

### Phase 1 - Evaluation Protocol and Repository Base

1. Create `src/{data,models,training,evaluation,visualization}`, `configs`, `scripts`, `tests`, and `outputs` (ignored except compact figures/tables).
2. Make group-aware, score-stratified train/validation/test splits (suggested 70/15/15) by QR/plot; fix and store split IDs. If data permits, use five group folds for confidence intervals.
3. Add configuration, seeds, checkpointing, early stopping, TensorBoard/CSV logging, environment file, and commands for audit/train/evaluate/infer.
4. Establish trivial references: training-set mean/median predictor and, optionally, frozen-feature ridge regression.

**Exit criterion:** one command rebuilds manifests/splits and tests prove no QR/plot leakage.

### Phase 2 - Required DINOv3 Baseline

1. Confirm DINOv3 weights/API access and input normalization. Start with the smallest feasible pretrained checkpoint and cache frozen embeddings to accelerate head experiments.
2. Train an MLP regression head on the 470 images. Use resize/crop plus mild photometric and geometric augmentation that preserves visible damage; avoid aggressive crops that remove plants or the frame.
3. Tune only a compact set: learning rate, head width/dropout, loss (Huber versus MSE), and image resolution. Select exclusively on validation MAE.
4. Evaluate once on the untouched test split. Export metrics, predictions CSV, learning curves, predicted-versus-true plot, residual histogram, worst/best examples, and inference speed.
5. Test transfer without retraining on progressively harder labelled sets: same BBCH/different images, different location/lighting, then older BBCH. Report each domain separately.

**Exit criterion:** reproducible checkpoint and baseline report sent to the supervisor, as requested in the brief.

### Phase 3 - Ranking-Based Weak Supervision

1. Build within-training-split pairs with absolute score gaps above `5`, `10`, and `20`; balance ordering and score ranges. Never form train pairs using validation/test images.
2. Compare regression-only, ranking-only, and joint loss `L = L_reg + lambda * L_rank`; tune temperature, margin, and `lambda` on validation data.
3. Add lower-confidence labelled images using ranking targets or disagreement-based weights. Compare high-quality-only versus expanded training data under the identical test split.
4. Measure MAE and ranking accuracy overall and by score gap/location. Use bootstrap confidence intervals and multiple seeds; keep the extension only if improvement is consistent.

**Exit criterion:** controlled ablation showing whether ranking supervision improves robustness beyond the baseline.

### Phase 4 - Plant-Aware Analysis and Agronomic Features (Stretch)

1. Use SAM/YOLO/RF-DETR to isolate plants and the metal-frame region; manually review a small stratified sample and quantify segmentation/detection quality.
2. Predict each plant's damage and combine plants using visible leaf area as weight, matching the project note. Compare this with whole-image scoring.
3. Prototype separate masks/detectors for shot holes and yellow/brown pitting. Compute counts and damaged area relative to estimated intact leaf area; flag edge damage as uncertain because missing leaf boundary is not directly observable.
4. Add plant count and BBCH/leaf-count estimates. Validate feature outputs on a small manually annotated test subset before presenting them as quantitative results.

**Exit criterion:** feature table plus overlay visualizations; clearly label unvalidated features as prototypes.

### Phase 5 - Resistance Ranking and Validation

1. Aggregate image scores into plot scores using the three QR-linked views; retain view-to-view variation as an uncertainty signal.
2. Normalize or model damage within each location/date/BBCH/experiment block so weather, growth stage, and pest pressure are not mistaken for genotype resistance.
3. Aggregate replicated plots for each genotype and report mean adjusted damage, confidence interval, sample count, and rank. A genotype is "more resistant" only when it has consistently lower damage with sufficient evidence.
4. Validate predicted genotype rankings against rankings obtained from manual scores using Spearman correlation, pairwise ordering accuracy, and top-/bottom-resistant overlap. Show sortable tables and representative damage overlays.

**Exit criterion:** defensible resistance leaderboard with uncertainty and traceability from genotype to plots to source images.

### Phase 6 - Packaging, Presentation, and Handover

1. Provide inference for one image/folder and a lightweight viewer showing image, score, confidence/uncertainty, overlays, and CSV export.
2. Freeze configs/checkpoints, rerun the final test evaluation, and document setup, data paths, commands, limitations, and model/data versions.
3. Prepare a 10-12 slide story: agricultural problem; data and noisy labels; leakage-safe protocol; DINOv3 baseline; ranking method; damage visualizations; genotype resistance ranking; qualitative failures; limitations; next steps.
4. Rehearse answers about label subjectivity, the 470-image choice, plot grouping, domain shift, metrics, compute, and why ranking is useful. Keep a prerecorded demo/figures as backup.

## 4. Experiment Matrix (Minimum)

| ID | Training data/objective | Purpose |
|---|---|---|
| E0 | Mean/median prediction | Sanity reference |
| E1 | Frozen DINOv3 + regression head, 470 HQ | Required baseline |
| E2 | E1 with augmentation/loss tuning | Strong baseline |
| E3 | 470 HQ, ranking only | Test ordinal supervision |
| E4 | 470 HQ, joint regression + ranking | Main proposed model |
| E5 | E4 + lower-confidence labels/weights | Test weak-label scaling |
| E6 | Best model on other locations/BBCH | Domain robustness |
| E7 | Plant-aware aggregation/features | Stretch goal |
| E8 | Plot/genotype aggregation versus manual ranking | Resistance-ranking validity |

## 5. Risks and Guardrails

- **Noisy ground truth:** report rater disagreement and uncertainty; do not overstate small metric gains.
- **Leakage:** group by QR/plot and inspect duplicates before splitting; three images of one plot are correlated.
- **Domain shift:** report results by location, lighting, date, and BBCH, not only a pooled score.
- **Small dataset/overfitting:** freeze backbone first, limit tuning, use grouped folds/multiple seeds, and preserve the test set.
- **Compute/access:** request Bender resources early; verify DINOv3 weight access immediately. Cached embeddings provide a fallback for baseline work.
- **Feature validity:** segmentation alone cannot recover eaten leaf edges reliably; validate on manual masks/counts and distinguish measured from inferred quantities.

## 6. Definition of Done

The project is complete when another student can recreate the split, train/evaluate the baseline and best extension, and reproduce both the damage results and genotype resistance ranking from documented commands. The presentation must include honest held-out comparisons, domain-shift and failure analysis, visible damage evidence, ranking uncertainty, a usable demo, and a clear distinction between validated results and exploratory feature extraction.
