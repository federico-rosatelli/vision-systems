# Report Task Tracker

Status of the three end-of-semester deliverables. Last updated **2026-09-27**.
Tags: `[ ]` open, `[IN PROGRESS]`, `[NEEDS VERIFICATION]`, `[x]` done and verified.

## Deadlines (from Luca Eichler's emails)

- **Report and video: end of September 2026** (3 days left as of 2026-09-27).
- **Repo access for Luca: end of the month** — done.

---

## Where things stand

- **Report:** full draft written, compiles to **12 pages** (hard cap) with no LaTeX warnings.
  Output: `report/main.pdf`. All numbers were re-run or reproduced on 2026-09-27.
- **Video:** not started.
- **Repo:** Luca informed about access; RF-DETR checkpoint not sent yet.

---

## TODO — manual, for us to do

### Report (before submitting)

- [x] **Matriculation numbers** added to `report/main.tex` (2026-09-27).
- [ ] **Read-through by both authors** (Emrullah + Federico). Check that the wording sounds like us,
      especially Conclusion → "Limitations" and "Recommendations for future data recording".
- [x] **Bib entries checked online (2026-09-27)**: `dinov3` matches arXiv:2508.10104 (Siméoni et al.,
      26 authors); `rfdetr` now cites the paper (Robinson et al., ICLR 2026, arXiv:2511.09554).
- [ ] **Decide the headline framing** (currently: area-weighted is the selected model, ranking loss is
      the improvement, uniform's better test result is reported openly in §4.3). Change only if
      both authors prefer a different framing — do not re-select on the test set.
- [ ] After any edit: rebuild (save in VS Code with LaTeX Workshop, or
      `cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main`) and confirm
      **≤ 12 pages**.
- [ ] **Submit** the PDF (and ask Luca whether he also wants the LaTeX sources).

### Video (3–10 min, for the Giessen collaborators)

- [ ] Make narrated slides (reuse report figures from `report/figures/`). Suggested outline, ~7 min:
  1. Problem + data (1 min): CSFB damage, 470 reliable images, raters correlate only r = 0.549.
  2. What we tried (2 min): classical CV → whole-image DINOv3 → plant-level MIL → + ranking loss.
     Figure 1 (pipeline), Table 2 (results).
  3. What did **not** work (1.5 min): whole-image baseline (chance-level ranking), shot-hole colour
     rules (0/40 usable), leaf counting (damage breaks it), RF-DETR (weak: mAP@50 3.5 %, too few
     labels), zero-shot transfer to Weilburger Grenze / DSV.
  4. Positive surprise (0.5 min): on images where JLU and GAU disagree, the model agrees with each
     rater better than they agree with each other.
  5. **Recommendations for recording data** (1.5 min) — the most useful part for Giessen:
     same frame + camera height so the frame fills the photo; detectable frame markers (coloured
     corners / ArUco); a small double-rated calibration set per site and date; record whether
     plants or the whole image were scored; score holes and pitting separately; annotate
     150–300 images per damage class for a detector.
     Figure 4 (the three recording setups) makes this point visually.
  6. Outlook (0.5 min): robust frame/plant detection (SAM), few-shot calibration per site, domain
     adaptation on the 8,946 unlabelled images, VLM weak labels.
- [ ] Record, check length (3–10 min), send to Luca.

### Repository

- [x] Luca informed about repo access (2026-09-27).
- [ ] Send the RF-DETR checkpoint separately
      (`outputs/rfdetr_hole_pitting/checkpoint_best_total.pth`, 116 MB, gitignored). Needed only for
      the webapp's live lesion detection; see `analyses/WEBAPP_PLAN.md` §3.
- [ ] Commit the new scripts and figures before sharing: `scripts/evaluate_generalization.py`,
      `scripts/train_joint_ranking.py`, `scripts/plot_report_scatter.py`, `report/` (chapters,
      figures, bib). Run outputs under `outputs/` stay local (large / gitignored).

---

## Optional — only if time is left (not needed for submission)

- [ ] Fix the webapp model list (`scripts/app.py`): `baseline_regression_*` runs don't exist
      (use `wholeimage_huber_seed42`), drop the duplicate `patch_joint_*`, add `joint_weighted_*`,
      rename "Rauischholzhausen" → "Weilburger Grenze". See `WEBAPP_PLAN.md` §2b.
- [ ] Rename `Rauischholzhausen_WG1` → `WeilburgerGrenze_WG1` in `src/evaluation/evaluate_ood.py`
      (label only; numbers unchanged).
- [ ] Remove or clearly mark the duplicate checkpoint `outputs/runs/patch_joint_seed42`.
- [ ] Ranking loss + uniform pooling (not tested; would take ~1 min on cached features). Only mention
      in the report if run with 3 seeds and evaluated on val first.

---

## Facts the report relies on (so nobody reintroduces a stale number)

Source of truth: `analyses/FINAL_PROJECT_RESULTS.md` (§4 baseline, §5–6b MIL + ranking + test,
§7 generalization, §8 classical CV). Key values:

| What | Value |
| :--- | :--- |
| Whole-image baseline, test | MAE 4.65, Spearman −0.11, pairwise 0.46 (constant median MAE 4.01) |
| Area-weighted MIL, 3 seeds, val / test | MAE 2.69 / 2.62, Spearman 0.853 / 0.763 |
| Uniform MIL, 3 seeds, val / test | MAE 2.73 / **2.40**, Spearman 0.838 / **0.808** |
| + ranking loss, 3 seeds, val / test | MAE 2.66 / 2.51, Spearman 0.859 / 0.781 |
| Selected model seed 42, test | MAE 2.60, Spearman 0.765, 95 % CI [0.62, 0.86] |
| Weilburger Grenze (WG1, n=906) | MAE 4.32 (constant 4.10), Spearman 0.03, frame found 59 % |
| DSV Trial 1 (n=897) | MAE 7.73 (constant 5.22), Spearman 0.26, frame found 0.1 % |
| GG raters disagree (n=218) | model vs JLU / GAU Spearman 0.50 / 0.55; JLU vs GAU 0.27 |
| Direct damage (40 images) | Spearman 0.64 (pitting 0.63, holes 0.20); holes 0/40 usable |
| RF-DETR | mAP@50 3.5 % after EXIF fix (0.4 % before) |

Do **not** cite: validation Spearman 0.855 as the held-out result; OOD Spearman ~0.20;
"Rauischholzhausen"; `patch_joint_*` results; the old `resistance_leaderboard.csv`; anything from
`data_manifest_split.csv` (leaky manifest).

---

## Completed (2026-09-27)

- [x] Report chapters written: abstract, introduction, related work, methodology, experiments,
      conclusion; 4 figures (pipeline TikZ, qualitative CV/RF-DETR, scatter, recording setups) and
      3 tables; 13 references; 12 pages.
- [x] Report framing agreed with the email thread: baseline fails → MIL (Luca's suggestion) → area
      weighting (his formula) → ranking loss (our chosen direction) → OOD as a main result (he asked)
      → RF-DETR side project (he recommended). Luca's label-definition comment used to discuss
      uniform vs. area weighting.
- [x] Ranking-based training evaluated properly: 3 seeds, same settings as the other MIL runs.
- [x] Whole-image baseline reproduced exactly (Huber, seed 42).
- [x] Extended generalization study incl. preprocessing diagnostics, strata, plot level, constant
      predictors and rater agreement (`scripts/evaluate_generalization.py`).
- [x] Methodology cross-checked against the code (HSV thresholds, baseline input, pair sampling).
- [x] All `analyses/*.md` files updated with the new results and corrections.
- [x] VS Code set up for LaTeX (LaTeX Workshop recipe without latexmk, auto-clean of aux files).
