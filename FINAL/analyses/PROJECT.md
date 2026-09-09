# CSFB Damage Quantification Project

## Objective

Estimate cabbage stem flea beetle damage in young rapeseed plants from field images and provide reproducible, biologically meaningful results for later genotype-resistance analysis.

The target is the percentage of visible leaf area affected by shot holes and yellow/brown pitting. Only plants inside the metal frame are included.

## Supervisor clarification

The supervisor clarified that plants should not receive equal weight. For plant instance \(i\), use

\[
w_i = \frac{A_i}{\sum_j A_j},
\]

where \(A_i\) is the number of pixels occupied by the plant mask. The image prediction is therefore

\[
\hat y = \sum_i w_i \hat y_i.
\]

This visible-area-weighted average is the primary aggregation rule. Uniform averaging is retained only as a diagnostic baseline because small false plant regions could otherwise have the same influence as real plants. Attention-based MIL may also be tested, but it must beat the fixed weighted method and its attention maps must be inspected for artifacts.

Individual plants remain unlabeled instances inside an image bag. The image-level score must not be copied to every plant as if it were a plant-level ground-truth label.

The supervisor also recommended testing a direct measurement:

\[
\text{damage percentage} =
\frac{\text{pixels classified as holes or pitting}}
     {\text{estimated total leaf pixels}}
\times 100.
\]

This direct measurement is an interpretable baseline, not yet validated ground truth.

## Data contract

- Authoritative manifest: `outputs/tables/baseline_manifest_split.csv`.
- Population: 470 unique BBCH10 images with consistent JLU and GAU scores.
- Fixed plot-group split: 331 train, 66 validation, and 73 test images.
- No filename, physical-image, or plot-group overlap is allowed between splits.
- Develop and select models using train and validation only. The 73-image test set is frozen.
- The source dataset is read-only. All derived artifacts belong under `outputs/`.

## Valid completed results

### Whole-image frozen-DINOv3 baseline

The complete image was resized to 224 x 224, which removes much of the small damage detail.

| Metric | Validation | Test |
|---|---:|---:|
| MAE | 5.1265 | 4.6502 |
| Spearman rho | 0.2712 | -0.1078 |
| Pearson r | 0.2391 | -0.0971 |

The model is worse than constant predictors on test MAE and does not learn useful test ordering.

### Plant-focused area-weighted model

Pipeline: frame interior -> high-resolution plant regions -> frozen DINOv3 features -> visible-area-weighted aggregation -> regression.

The corrected formulation predicts a score for every plant instance and then averages the scores using normalized plant-mask area. Across seeds 42, 43, and 44 it achieves validation MAE 2.6932 +/- 0.0232 and Spearman rho 0.8532 +/- 0.0022. It outperforms uniform score averaging, ABMIL, and gated ABMIL. The selected validation run is `aggregation_weighted_seed42`.

### Biological damage features

A corrected classical computer-vision audit separately estimates enclosed holes and pitting using valid plant-region crop coordinates. On a balanced 40-image train/validation set, its preliminary direct damage percentage reaches Spearman rho 0.6425 and Pearson r 0.4564 against the expert score. Manual review found the leaf masks and pitting regions mainly acceptable but rejected the shot-hole detections as unreliable. The combined direct percentage is therefore not validated. Leaf area and pitting remain exploratory features while shot-hole detection requires improvement; edge damage also remains unmeasured.

A soil-aware hole classifier was compared with the old brightness-only rule on 20 difficult train/validation images. Manual inspection showed that it still confused apparent holes with pitting and behaved inconsistently across soil conditions, so it is rejected. Further threshold tuning is not defensible without pixel-level labels. Reliable separate hole and pitting measurements now require a small manually annotated segmentation set.

## Results that must not be used as final evidence

The later cached ABMIL experiment used `outputs/tables/data_manifest_split.csv`, which is not the fixed 470-image manifest. It contains duplicated physical images across splits and records with unknown plot groups. The resulting 612-sample MIL test metrics are contaminated and invalid for model comparison.

The existing OOD results on Rauischholzhausen and DSV are preliminary. They show weak rank correlations of about 0.20 and a large DSV calibration error, but the referenced MIL checkpoint is missing and its training provenance is affected by the invalid manifest. These experiments must be repeated with a clean, frozen model.

The current resistance leaderboard must also be regenerated. It includes an `unknown` group and does not yet provide a valid replicated genotype comparison.

## Current decision

Use visible plant-mask pixel area as the primary aggregation weight. The data/cache contract is repaired, the 40-image mask audit passed, and the three-seed aggregation experiment selected area-weighted plant-score pooling. Both classical shot-hole rules failed manual review, so the combined direct percentage cannot be used. The next step for separate biological features is manual hole/pitting segmentation annotation; the trained area-weighted scoring model can proceed independently to clean OOD evaluation.

The compact plant-weight audit evidence is stored in Git, while its per-image masks and overlays are ignored and regenerated locally using the documented command. Regeneration preserves the recorded manual review.

See `analyses/CONTINUATION_PLAN.md` for the implementation order.
