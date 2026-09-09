# CSFB Continuation Plan

## Fixed decisions

- Analyze only plants inside the metal frame.
- Preserve high-resolution plant regions; do not downsize the full field image as the main input.
- Use the fraction of total plant-mask pixels as each plant's aggregation weight.
- Treat plants as unlabeled instances within an image bag; do not copy the image score to each plant as an independent label.
- Treat uniform averaging only as a diagnostic baseline.
- Treat learned MIL attention as experimental and inspect its attention maps.
- Evaluate direct damaged-leaf-area percentage as an interpretable alternative.
- Use only the fixed 470-image grouped manifest for in-domain development.
- Keep the 73-image test split closed during model selection.

## Immediate work

### 1. Repair the data and cache contract

Status: complete.

Completed safeguards:

- Patch and MIL configurations use `outputs/tables/baseline_manifest_split.csv`.
- Training, evaluation, and caching reject an invalid frozen manifest.
- Caches record the manifest hash and preprocessing configuration.
- Cache loading rejects a mismatched manifest or filename set.
- Cached datasets apply the requested quality filter.
- The test suite passes 37 tests.

The contaminated cache is preserved as `outputs/cache/dinov3_bags.INVALID_LEAKED.pt`. A clean, provenance-checked cache containing exactly 470 bags was regenerated as `outputs/cache/dinov3_bags.pt`.

Embedding caches are generated locally and are not stored in Git. After cloning or pulling the repository, each team member should download the authorized DINOv3 weights as described in `analyses/DINOV3_SETUP.md`, then run:

```bash
cd vision-systems/FINAL
source .venv/bin/activate
python main.py --config configs/config_mil_cache.json
```

The command validates the frozen manifest before feature extraction and writes `outputs/cache/dinov3_bags.pt`. A valid cache contains exactly 470 bags and records the manifest hash and preprocessing settings. Training and evaluation reject the cache if it does not match `outputs/tables/baseline_manifest_split.csv`.

Never copy or use `outputs/cache/dinov3_bags.INVALID_LEAKED.pt`; it is an invalid historical artifact and may be deleted locally.

### 2. Validate plant masks as area weights

Status: complete. The 40-image manual review was approved.

Artifacts:

- Configuration: `configs/plant_weight_audit.json`
- Generator: `src/preprocessing/plant_weight_audit.py`
- Review table: `outputs/plant_weight_audit/plant_weight_audit.csv`
- Summary: `outputs/plant_weight_audit/plant_weight_audit_summary.json`
- Contact sheet: `outputs/plant_weight_audit/plant_weight_contact_sheet.jpg`

The review CSV, summary JSON, and contact sheet are small and should be committed so the team shares the audit evidence. The per-image files under `outputs/plant_weight_audit/masks/` and `outputs/plant_weight_audit/overlays/` are generated locally and ignored by Git because together they are about 47 MB.

After cloning or pulling, regenerate the ignored per-image files with:

```bash
cd vision-systems/FINAL
source .venv/bin/activate
python -m src.preprocessing.plant_weight_audit \
  --config configs/plant_weight_audit.json
```

Regeneration preserves existing manual-review columns in `plant_weight_audit.csv` by matching filenames. It recreates the masks, overlays, contact sheet, and automatic summary fields from the fixed manifest and configuration.

The set contains 20 train and 20 validation images, ten images from each of four damage ranges, and no test images. All 40 frames were detected, all numerical area weights sum to one, and the reviewer marked all 40 masks and area weights as usable. The usable rate is 100%, so the 90% exit criterion is met.

Review a stratified train/validation sample covering different scores, lighting, soil, and plant sizes. For each image record:

- whether all in-frame plants are detected;
- false plant regions;
- merged or split plants;
- missing green/yellow/brown leaf tissue;
- frame-crop failure;
- whether mask area is a reasonable approximation of visible plant area.

The current green mask was approved for finding patches, but not as a complete leaf-area mask. In particular, damaged non-green tissue must not disappear from the denominator without being accounted for.

Future reviewers can reproduce the summary after editing the review table with:

```bash
python -m src.preprocessing.plant_weight_audit \
  --config configs/plant_weight_audit.json \
  --summarize-review
```

### 3. Run a controlled aggregation experiment

Train and select models using train and validation only. Keep the frozen DINOv3 representation and preprocessing comparable.

Compare:

1. uniform mean pooling, as an artifact-sensitivity baseline;
2. visible-area-weighted pooling, the primary method;
3. ABMIL attention pooling;
4. gated ABMIL attention pooling.

Run at least three seeds. Report MAE, RMSE, Pearson, Spearman, prediction spread, score-bin errors, pairwise accuracy, and constant baselines. For attention models, save instance weights and overlays to check whether the model focuses on plants rather than soil or segmentation errors.

Primary selection rule: lowest mean validation MAE across seeds, provided that Spearman and prediction spread do not materially degrade. Prefer visible-area weighting when performance is comparable.

### 4. Evaluate direct damaged-leaf-area percentage

For each plant and image, compute and report separately:

- estimated total leaf area;
- shot-hole count and area;
- yellow/brown-pitting count and area;
- combined damaged area percentage;
- pitting-to-hole ratio.

Validate these outputs on manually reviewed masks before interpreting them biologically. Compare the direct percentage with expert scores using MAE after validation-only calibration, Pearson, Spearman, and error plots. Clearly report that edge bites remain uncertain unless a validated edge-reconstruction method is added.

### 5. Repeat the OOD study

Use the single model selected and frozen in Step 3. Save per-image predictions and preprocessing diagnostics for Rauischholzhausen and DSV.

Report overall results and results stratified by location, BBCH stage, brightness/lighting, soil appearance, score range, and preprocessing success. Compare against constant predictors and the direct damaged-area baseline. A small visual audit should identify whether failures arise from frame detection, plant masks, plant age, illumination, or the learned representation.

### 6. Perform genotype analysis

Only after the clean in-domain and OOD evaluations are complete:

1. aggregate image predictions to plot level;
2. join verified genotype metadata;
3. compare genotypes only within compatible location, experiment, date, and BBCH conditions;
4. require biological replication and report uncertainty;
5. compare predicted rankings with expert-score rankings.

Do not include unknown groups in a resistance leaderboard.

## Completed supporting work

- Leakage-safe 470-image baseline manifest and split.
- Whole-image frozen-DINOv3 baseline.
- Frame detection and a manually approved 30-image crop audit.
- Plant-region proposal and a manually approved 30-image patch audit.
- Variable-length plant-patch dataset.
- Area-weighted plant-focused regression implementation.
- Uniform, ABMIL, and gated-ABMIL model implementations.
- Preliminary hole and pitting feature extraction.
- OOD evaluation code.

## Completion criteria

The project is ready for final presentation when another student can reproduce the selected model and all reported tables from the fixed manifest; every cache and checkpoint has verified provenance; no split leakage exists; plant-area weights have been visually validated; OOD limitations are reported; and genotype rankings contain only properly matched, replicated data.
