# Pre-Report Fixes Plan

Scope: get the pipeline and results into a trustworthy, reproducible state before
starting the written report. Report writing is explicitly out of scope for this
plan (tracked separately in `report_tasks.md`).

## Context

A live `pytest` run (2026-09-21) shows the suite is currently red:

```
3 failed, 45 passed, 0 skipped
```

This contradicts an earlier stale reading of the `.pytest_cache` ("44 passed / 4
skipped"). All three failures share one root cause.

## 1. Fix `create_splits` dtype bug

Status: complete (2026-09-21).

- File: `src/data/make_splits.py:50`.
- Cause: `group_stats['score_bin'].values` now returns a pyarrow-backed array
  (current pandas version) instead of a plain numpy array. Passing it to
  sklearn's `train_test_split(..., stratify=group_bins)` raises
  `TypeError: only integer scalar arrays can be converted to a scalar index`
  inside `_safe_indexing`.
- Both code paths are affected: the stratified split throws (caught by the
  `except ValueError` and would be masked in normal-sized data), and the
  **fallback random-split path also crashes** on the same malformed array,
  so `create_splits()` is completely broken in the current environment, not
  just degraded.
- Failing tests, all downstream of this function:
  - `tests/test_config.py::test_config_loading`
  - `tests/test_data.py::TestDataPipeline::test_create_splits_leakage`
  - `tests/test_pipeline.py::test_full_pipeline`
- Fix: convert `group_bins` (and any other categorical `.values` passed into
  sklearn splitters) to a plain numpy array, e.g.
  `group_stats['score_bin'].astype(int).to_numpy()`.
- Acceptance: `pytest tests/ -q` reports `0 failed`.
- Done: `src/data/make_splits.py:41-42` now uses `.to_numpy(...)` instead of
  `.values`. `pytest tests/ -q` → `48 passed`.
- Side finding: `src/evaluation/evaluate_ood.py` was also missing an
  `argparse` import (dead code, only hit when run as `__main__` rather than
  via `main.py`); fixed alongside this.

## 2. Confirm existing splits/results are unaffected

Status: complete (2026-09-21).

- Finding: the frozen 470-image split (`outputs/tables/baseline_manifest_split.csv`)
  is actually produced by a **separate** deterministic pure-Python script,
  `src/data/make_baseline_splits.py` (no pandas/sklearn involved), not by the
  buggy `make_splits.py`. The bug in (1) never affected it.
- Verified: re-ran `python -m src.data.make_baseline_splits --seed 42` against
  `outputs/tables/baseline_manifest.csv` and diffed the output against the
  existing `baseline_manifest_split.csv` — **byte-for-byte identical**.
- No regeneration needed; existing caches/checkpoints remain valid.

## 3. Re-run the OOD evaluation

Status: complete (2026-09-21).

- Ran `python main.py --config configs/config_mil_eval.json` (action
  `evaluate_ood`) with `outputs/runs/mil_weighted_seed42/checkpoints/best_model.pth`.
  Manifest SHA-256 matches the current frozen `baseline_manifest_split.csv`
  (`7048425afdb49fcd0fdf94c3c703b012bde008652e9ea2dcb08c7e7b9d3f24a8`) —
  provenance-clean.
- New results (`outputs/tables/ood_evaluation_results.csv` /
  `_summary.json`), superseding the stale preliminary numbers:
  - Rauischholzhausen WG1 (N=906): MAE 4.33%, RMSE 6.27%, Pearson 0.113,
    Spearman 0.032 (previously 4.29% / 5.83% / 0.293 / 0.199 — rank
    correlation is now essentially zero, weaker than previously reported).
  - DSV Trial 1 (N=897): MAE 8.11%, RMSE 9.49%, Pearson 0.309, Spearman 0.270
    (previously 15.60% / 17.24% / 0.207 / 0.193 — MAE and rank correlation
    both improved substantially).
- Conclusion unchanged: the model generalizes poorly to unseen field trials.
  `analyses/FINAL_PROJECT_RESULTS.md` §7 updated with the confirmed numbers.

## 4. Re-verify headline MIL numbers reproduce

Status: complete (2026-09-21) — found and fixed a real reporting error.

- `outputs/tables/aggregation_validation_summary.csv` / `_results.csv`
  already contained the 3-seed aggregation comparison; numbers match
  `analyses/FINAL_PROJECT_RESULTS.md` §5 exactly. Weighted aggregation still
  ranks first. No retraining needed for this part.
- However: `aggregation_weighted_seed42` had **never been evaluated on the
  closed 73-image test set** (no `outputs/runs/aggregation_weighted_seed42/tables/`
  directory existed). §6 of `FINAL_PROJECT_RESULTS.md` ("Selected Production
  Model Performance") was titled `mil_weighted_seed42` but its numbers
  (MAE 2.6670, Spearman 0.8553) are actually `aggregation_weighted_seed42`'s
  **validation** metrics, mislabeled under the wrong checkpoint name.
- Ran `main.py --action evaluate_mil` against `aggregation_weighted_seed42`
  on the real closed test set (N=73): **MAE 2.60%, RMSE 3.78%, Pearson 0.763,
  Spearman 0.765, pairwise-gap5 92.5%** — noticeably lower Spearman/Pearson
  than the validation numbers (0.855 / 0.859) that were being cited as the
  headline result. MAE/RMSE hold up fine; rank correlation does not.
- Cross-check: independently re-evaluated `mil_weighted_seed42` (a separate
  training run, same config/seed, trained by a teammate) on the same test
  set — MAE 2.59%, Spearman 0.764. The two independently trained checkpoints
  agree closely, which is a good reproducibility signal, but both confirm the
  val→test rank-correlation gap is real, not a one-off fluke.
- Fixed: `analyses/FINAL_PROJECT_RESULTS.md` §6 now correctly attributes the
  validation numbers, and adds the actual test-set table with both
  checkpoints. **The report should cite the test-set numbers (Spearman
  ~0.765), not the validation numbers (~0.855), as the primary result.**

## 5. Decide scope on genotype/resistance leaderboard

Status: investigated (2026-09-21) — blocked on a data-availability limit, needs a decision.

- Confirmed `outputs/tables/resistance_leaderboard.csv` was built from a
  different, larger/messier manifest than the frozen 470-image set (it
  contains an "unknown" `plot_group` row with 443 images). The frozen
  470-image manifest itself has **no** unknown genotypes (0/470).
- However, checked genotype replication within the frozen manifest
  (`outputs/tables/baseline_manifest_split.csv`): 217 distinct genotypes
  across 250 plot groups, and **only 7 of 217 genotypes (3%) have 2 or more
  plot-group replicates** — the other 210 are single-plot singletons. The
  73-image test split alone spans 36 genotypes with 1-2 images each.
- Implication: a statistically meaningful *genotype-level* resistance
  ranking (i.e. one that supports a "genotype A is more resistant than
  genotype B" claim) is not really supportable from the current 470-image
  gold-standard set — there isn't enough replication. A larger, noisier pool
  (e.g. the full ~2,747-image scored set, or ~8,946 total) would have more
  replication but weaker/less-calibrated scores.
- **Needs a decision, not a silent fix:**
  - (a) Build a proper plot-level (not genotype-level) leaderboard from the
    frozen 470-image set, filtered to real plot groups, and be explicit that
    it's plot-level, not a validated genotype comparison; or
  - (b) Attempt genotype-level aggregation only on genotypes with ≥2 plot
    replicates (≈7 genotypes) as a small, caveated illustration; or
  - (c) Report genotype/resistance ranking as future work and drop it from
    the report's results entirely, given the replication limit.
- **Resolved (2026-09-21):** built `scripts/build_resistance_leaderboard.py`,
  which runs `aggregation_weighted_seed42` inference over all 470 cached
  bags (no held-out restriction — this is a descriptive tool, not a model
  evaluation) and produces:
  - `outputs/tables/plot_resistance_leaderboard.csv` — 250 plot groups, 470
    images, zero "unknown" rows, ranked by predicted damage. Plot-level
    pred-vs-true agreement: Pearson 0.832, Spearman 0.830 (sanity check, not
    a held-out metric since train/val plots were seen during training).
  - `outputs/tables/genotype_resistance_subset.csv` — the 7 genotypes with
    ≥2 plot replicates (Express 617, WINFRED, Smaragd, RESZN/H048,
    ALLESANDRO KWS, R31, Taisetsu; 2-17 plots each), explicitly labeled as a
    small, caveated illustration, not a validated genotype comparison.
  - Old `outputs/tables/resistance_leaderboard.csv` (built from a different,
    unfrozen manifest containing an invalid "unknown" plot group of 443
    images) is superseded and should not be cited in the report.

## 6. Example figures for negative-result sections

Status: complete (2026-09-21).

- Added `scripts/render_rfdetr_hole_pitting_examples.py`, which renders
  side-by-side ground-truth vs. predicted-box figures for the rejected
  RF-DETR detector. Output: `outputs/rfdetr_hole_pitting/example_figures/`
  (2 images, ~1.1 MB total). Example: `20251021_132633_1.jpg` has 1 GT box
  vs. 166 predicted boxes scattered over soil texture, unrelated to the
  leaf; `20251021_122353_12.jpg` has 30 GT boxes vs. 204 predictions
  clustered on the frame crossbar. Both make the near-zero mAP result
  (0.004) immediately legible as a figure, not just a number.
  Note: the script defaults to `threshold=0.05`, not the `0.5` listed in
  `configs/rfdetr_hole_pitting_eval.json` — 0.05 is what actually reproduces
  the `pred_box_count` values already recorded in
  `outputs/rfdetr_hole_pitting/rfdetr_hole_pitting_review.csv` with the
  current `rfdetr` package version.
- Classical direct-damage audit already has a usable example without new
  code: `outputs/direct_damage_audit/overlays/20251021_120939_damage.jpg`
  shows the visual gap directly (expert score 20.25% vs. classical
  `direct=0.726%`), i.e. the classical shot-hole/pitting rule detects
  essentially nothing on an image an expert rated as one-fifth damaged.
- Added `!outputs/rfdetr_hole_pitting/example_figures/` to `.gitignore` so
  these renders survive the blanket `*.jpg` ignore rule.

## Out of scope for this plan

- Writing any `report/chapters/*.tex` content.
- RF-DETR hole/pitting detector and classical direct-damage-%: both already
  investigated and rejected; no further work planned here (to be written up
  as negative/exploratory results once report writing starts).
