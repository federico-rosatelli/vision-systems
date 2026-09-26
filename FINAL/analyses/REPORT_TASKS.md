# Report Task Tracker

This file tracks the progress of the final project report and the two other end-of-semester
deliverables (video, code/repo access). Read this whole file before starting any task.

## Deadlines (confirmed from Luca Eichler's emails)

- **Report and video: due end of September 2026.** Today is 2026-09-26 — this is imminent.
- **Repo access / code sharing with Luca: due end of the month.** Check now whether Luca
  (eichler@ais.uni-bonn.de) already has access to this repository; if not, add him or send the
  code today, independent of the report/video work below.

## Deliverable 1: Written report

- Format: **LaTeX**, English, structure similar to the basic Overleaf template Luca linked
  (https://www.overleaf.com/latex/templates/basic-template/wswqstsqbzbz) — plain `article`
  class, title/author/date, `\tableofcontents`, numbered sections, bibliography. Do not deviate
  from this significantly (no custom class, no multi-column layout, no heavy theming).
  `report/main.tex` already matches this (11pt article, geometry, hyperref, amsmath) — keep it.
- **Length: hard cap 12 pages** for the compiled PDF. Luca's first mail said 15 was already long;
  a later mail lowered the cap to 12. Treat 12 as the binding number. There is no stated
  appendix exception, so tables/figures count against the limit — budget for this from the
  start:
  - Keep Related Work / Background short (roughly half a page to a page); it is background, not
    a core contribution.
  - Prefer 1 consolidated results table per subsection over several small ones (see per-chapter
    notes below for which tables to merge).
  - Do not include every classical-CV contact sheet; pick 1-2 representative qualitative figures
    per pipeline stage (see 6.6 below).
- Update the tag on each task below as you work (tag definitions unchanged from before):
  `[ ]` not started, `[IN PROGRESS]`, `[NEEDS VERIFICATION]`, `[x]` complete and verified.

### Source-of-truth documents (read these, don't re-derive numbers)

- `analyses/PROJECT.md` — objective, supervisor's aggregation-weight formula, current decision,
  and which results are valid vs. must-not-use.
- `analyses/FINAL_PROJECT_RESULTS.md` — the actual numbers to cite (whole-image baseline,
  3-seed aggregation comparison, selected model val **and** test metrics, OOD benchmark,
  classical CV audit table, RF-DETR). This is the primary numbers source for Chapter 6.
- `analyses/CONTINUATION_PLAN.md` — the project's decision log and reasoning trail; useful for
  Methodology (why each choice was made) and Conclusion (what's actually finished vs. not).
- `analyses/PRE_REPORT_FIXES_PLAN.md` — records two corrections made after the fact: (a) which
  checkpoint the headline validation numbers actually belong to, and (b) that the frozen
  73-image **test** set shows a real val→test rank-correlation drop. **Do not report the
  validation Spearman (0.855) as if it were the held-out result** — see the Chapter 6 note below.
- `analyses/HOLE_PITTING_ANNOTATION_PLAN.md` — the RF-DETR annotation effort, the EXIF-orientation
  bug that initially made it look like a total failure, and the post-fix weak-positive result.
  Good source for a "check for pipeline bugs before blaming data scarcity" methodology remark.
- `Lab Vision Systems_ Lab Project.pdf` (project brief) and Luca's emails — problem framing,
  BBCH growth stages, the three original proposed directions (ranking-based training / few-shot
  VLM labeling / domain adaptation), and the visible-area weighting rule.

### Two content points from Luca's emails that must be addressed explicitly

1. **Visual score composition.** Luca clarified the JLU/GAU visual scores fold in *both*
   shot-holing and pitting, and are most likely a per-plant subjective score averaged over all
   plants in the image. Use this in Related Work/limitations to explain label noise/subjectivity
   — it connects directly to the project brief's own JLU-vs-GAU correlation plot (r ≈ 0.549,
   `Lab Vision Systems_ Lab Project.pdf` slide 9) and to why a single scalar target is a fairly
   coarse supervision signal for MIL.
2. **Ranking-based training.** This was one of the three original directions proposed in the
   project brief (pairwise `MarginRankingLoss`, "do we need perfectly labelled scores?").
   It **was implemented**: `src/models/losses.py::JointRankingRegressionLoss` (Huber regression
   + pairwise margin-ranking term), trained as `patch_joint_seed42` and
   `patch_joint_sampled_seed42` (see `outputs/runs/patch_joint*`). However, unlike the other
   runs, **these two checkpoints were never evaluated with the standard val/test MAE / Pearson /
   Spearman / pairwise-ranking-accuracy protocol** — only per-epoch training-log curves exist.
   Two options, per Luca's mail ("mention this and show the results, or use it as a future
   outlook if there weren't any results"):
   - **Preferred if time allows:** run `evaluate_mil` (or the equivalent evaluation path) against
     `outputs/runs/patch_joint_seed42/checkpoints/best_model.pth` and
     `patch_joint_sampled_seed42/checkpoints/best_model.pth` on the same frozen val/test split,
     and report the resulting table next to the aggregation comparison in 6.4.
   - **If time does not allow:** say explicitly in Methodology/Future Work that ranking-based
     training was implemented and trained but not yet evaluated to the same standard as the
     weighted/ABMIL/gated-ABMIL runs, and name it as the concrete next experiment. Do not silently
     drop it — it is one of the three directions the supervisor explicitly asked each team to
     touch.

### Chapter-by-chapter tasks

- [ ] **1. Project Setup** — `report/chapters/` and `main.tex` already scaffolded; no action
  needed unless the template needs adjusting for the 12-page limit (e.g. tighter margins are
  already set via `geometry`).

- [ ] **2. Abstract** — `report/chapters/00_abstract.tex`
  - One paragraph: problem (CSFB damage scoring on BBCH10 rapeseed for genotype resistance
    breeding, Res4StRes project), the full pipeline (classical-CV audits → whole-image baseline →
    area-weighted MIL → OOD evaluation), and the headline result: MIL area-weighted pooling beats
    uniform pooling and attention MIL, reaches test MAE 2.60% / Spearman 0.765, but generalizes
    poorly zero-shot to unseen field trials (Rauischholzhausen, DSV).

- [ ] **3. Introduction** — `report/chapters/01_introduction.tex`
  - [ ] Agricultural context: CSFB damage on young oilseed rape, Res4StRes multi-institution
    project, why early (BBCH10-11) damage scoring matters for resistance breeding.
  - [ ] Problem framing: subjective, sparse expert labels (470 "high-quality" of 8946 total
    images; JLU/GAU inter-rater correlation ≈ 0.549) → why this is a weakly-supervised MIL
    problem, not ordinary supervised regression.
  - [ ] Scope: classical CV heuristics, whole-image DL baseline, and MIL, all evaluated in- and
    out-of-distribution.
  - [ ] Contributions and report roadmap.

- [ ] **4. Related Work / Background** — `report/chapters/02_related_work.tex` (keep short, see
  page-budget note above)
  - [ ] Classical CV in agriculture (color/threshold-based leaf and lesion segmentation).
  - [ ] Whole-image regression/classification for phenotyping, and its resolution-loss failure
    mode motivating patch/MIL approaches.
  - [ ] MIL basics (bag/instance framing, attention pooling, Ilse et al. ABMIL).
  - [ ] OOD generalization in agricultural vision (domain shift across fields/seasons/lighting).

- [ ] **5. Methodology** — `report/chapters/03_methodology.tex`
  - [ ] **5.1 Data curation & frame extraction:** metal-frame detection/cropping, the fixed
    470-image leakage-safe manifest (`outputs/tables/baseline_manifest_split.csv`, 331/66/73
    plot-group split), why the 73-image test split stayed closed during development.
  - [ ] **5.2 Classical CV:** plant-region proposal (adaptive HSV), plant-mask area weighting,
    the direct damage measurement (holes+pitting / leaf pixels) and why its shot-hole component
    was rejected on manual review; mention the RF-DETR hole/pitting detector as a separate,
    exploratory detection attempt (annotation set, EXIF bug, weak-positive result).
  - [ ] **5.3 Whole-image baseline:** frozen DINOv3 + regression head on 224×224 downsampled
    frames.
  - [ ] **5.4 MIL:** frozen DINOv3 features per plant patch, supervisor's area-weighted
    aggregation formula \(w_i = A_i/\sum_j A_j\), uniform/ABMIL/gated-ABMIL as comparisons, and
    the ranking-based joint loss (`JointRankingRegressionLoss`) as the third direction — see the
    ranking-based-training note above for what to say about its evaluation status.
  - [ ] **5.5 OOD framework:** zero-shot evaluation protocol on Rauischholzhausen WG1 and DSV
    Trial 1, no fine-tuning on either.

- [ ] **6. Experiments & Results** — `report/chapters/04_experiments.tex`
  - [ ] **6.1 Setup:** dataset splits, hyperparameters, metrics (MAE, RMSE, Pearson, Spearman,
    pairwise gap-5 ranking accuracy).
  - [ ] **6.2 Classical CV audits:** one consolidated table (FINAL_PROJECT_RESULTS.md §3 and §8
    merged) covering frame audit, plant-region audit, plant-weight audit (all approved),
    direct-damage / hole-detection-comparison (rejected for holes, leaf+pitting acceptable),
    RF-DETR (weak positive post EXIF-bug-fix, mAP@50 3.47%), and the rejected
    leaf/cotyledon-counting attempt.
  - [ ] **6.3 Whole-image baseline:** MAE/Pearson/Spearman on val and test
    (FINAL_PROJECT_RESULTS.md §4) — note the negative test correlation and that this motivated
    the move to MIL.
  - [ ] **6.4 MIL results:** the 3-seed, 4-method aggregation comparison table (§5), then the
    selected `aggregation_weighted_seed42` model's **validation and test** metrics in one table
    (§6) — **cite the test-set Spearman (~0.765) as the primary generalization number, with the
    bootstrap 95% CI [0.62, 0.86] stated so the val→test gap (0.855→0.765) reads as expected
    small-sample noise, not an unexplained drop.** Add the ranking-based (`patch_joint_*`)
    results here too, per the note above (evaluated numbers if produced in time, else marked as
    trained-but-not-yet-evaluated).
  - [ ] **6.5 OOD generalization:** the confirmed (2026-09-21) numbers only — Rauischholzhausen
    WG1 MAE 4.33%/Spearman 0.032 (N=906), DSV Trial 1 MAE 8.11%/Spearman 0.270 (N=897). Do not
    cite the older ~0.20 Spearman numbers; they are stale (see PROJECT.md and
    PRE_REPORT_FIXES_PLAN.md §3).
  - [ ] **6.6 Qualitative analysis:** a handful of representative figures, not full contact
    sheets — e.g. one attention-map example, one classical-CV segmentation overlay, one
    prediction-vs-true scatter, and the before/after RF-DETR EXIF-bug figure
    (`outputs/rfdetr_hole_pitting/example_figures/`).
  - [ ] Optional, only if page budget allows: one short paragraph + table on the plot-level
    resistance leaderboard (`outputs/tables/plot_resistance_leaderboard.csv`) and the 7-genotype
    caveated subset (`genotype_resistance_subset.csv`), explicitly labeled as illustrative, not a
    validated genotype comparison (only 7/217 genotypes have ≥2 plot-group replicates).

- [ ] **7. Conclusion & Future Work** — `report/chapters/05_conclusion.tex`
  - [ ] Comparative summary: classical CV (interpretable but fragile/rejected components) vs.
    whole-image (fails, loses detail) vs. MIL (best, area-weighting beats attention pooling).
  - [ ] Limitations: label subjectivity/sparsity, weak OOD generalization, small hole/pitting
    annotation set, genotype replication limit.
  - [ ] Future work: finish ranking-based-training evaluation if not done; grow the hole/pitting
    annotation set (~150-300 images/class per HOLE_PITTING_ANNOTATION_PLAN.md's estimate);
    revisit domain adaptation / few-shot VLM labeling (the two directions not pursued in depth);
    stratified OOD failure analysis (by location/BBCH/lighting/soil) that
    CONTINUATION_PLAN.md flags as not yet done.

- [ ] **8. Final Review**
  - [ ] Compile and confirm the PDF is **≤ 12 pages**.
  - [ ] Fix formatting/typos/citations.
  - [ ] Cross-check every cited number against `analyses/FINAL_PROJECT_RESULTS.md` /
    `analyses/PROJECT.md` one more time — several numbers in this project were corrected after
    initial reporting (see PRE_REPORT_FIXES_PLAN.md); don't reintroduce a stale one.

## Deliverable 2: Video (for the University of Giessen collaborators)

- Length: **3-10 minutes**. Format is flexible — narrated slides is fine, doesn't need to be a
  live demo.
- Must cover:
  1. Approaches tried (classical CV, whole-image baseline, MIL, ranking-based training if
     evaluated) and the results obtained.
  2. What did **not** work, stated plainly (whole-image baseline, shot-hole classical detectors,
     leaf/cotyledon counting, weak OOD generalization).
  3. Recommendations/tips for recording data in the future to make learning and OOD
     generalization more feasible (e.g. consistent lighting/frame positioning, more
     geographically/temporally diverse labeled examples, larger hole/pitting annotation set).
  4. Outlook on how the method could be improved further.
- Can reuse the report's figures/tables directly; no separate deck is required if the report
  slides/figures already tell the story.

## Deliverable 3: Repository access

- [ ] Confirm Luca Eichler has access to this repository (or has been sent the code) — due end
  of month, separate from the report/video deadline.
