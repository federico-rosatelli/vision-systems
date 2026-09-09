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

The recorded validation result is MAE 2.8104 and Spearman rho 0.8403. This is strong evidence that preserving plant-level detail is useful. The run metadata uses the correct 470-image manifest. Before making further test claims, its checkpoint and prediction artifacts must be restored and verified against that manifest.

### Biological damage features

A classical computer-vision prototype separately estimates enclosed holes and pitting. On a stratified 30-image audit it reached Spearman rho 0.6522 against the expert score. This result is exploratory because the masks lack pixel-level manual validation, edge damage is missed, and its absolute scale differs from the expert scores.

## Results that must not be used as final evidence

The later cached ABMIL experiment used `outputs/tables/data_manifest_split.csv`, which is not the fixed 470-image manifest. It contains duplicated physical images across splits and records with unknown plot groups. The resulting 612-sample MIL test metrics are contaminated and invalid for model comparison.

The existing OOD results on Rauischholzhausen and DSV are preliminary. They show weak rank correlations of about 0.20 and a large DSV calibration error, but the referenced MIL checkpoint is missing and its training provenance is affected by the invalid manifest. These experiments must be repeated with a clean, frozen model.

The current resistance leaderboard must also be regenerated. It includes an `unknown` group and does not yet provide a valid replicated genotype comparison.

## Current decision

Use visible plant-mask pixel area as the primary aggregation weight. The data/cache contract is repaired and a clean 470-bag cache can be generated reproducibly from the frozen manifest. Generated caches are not stored in Git; each team member creates one locally using the command in `analyses/CONTINUATION_PLAN.md`. The balanced 40-image plant-weight audit passed manual review: all masks and weights were marked usable. The next step is the controlled aggregation experiment. A clean OOD study follows model selection, and genotype ranking comes last.

The compact plant-weight audit evidence is stored in Git, while its per-image masks and overlays are ignored and regenerated locally using the documented command. Regeneration preserves the recorded manual review.

See `analyses/CONTINUATION_PLAN.md` for the implementation order.
