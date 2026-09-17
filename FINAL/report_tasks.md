# Report Task Tracker

This file tracks the progress of the final project report. The report must be a maximum of 15 pages and written entirely in English.
Please read the instructions below before starting.

## Instructions for Agents

When working on this report, you must update the tags in this file to reflect the current status of each section.

### Tag Definitions:
- `[ ]` : **Not Started**. No one is currently working on this task.
- `[IN PROGRESS]` : **In Progress**. An agent is currently actively working on this task. When you start a task, update the tag to this.
- `[NEEDS VERIFICATION]` : **Needs Verification**. The agent has finished their part, but it requires human review, missing points need to be addressed, or the agent is unsure how to proceed.
- `[x]` : **Completed**. The task is fully complete and verified.

### General Workflow:
1. Claim a task by changing its tag from `[ ]` to `[IN PROGRESS]`.
2. Edit the corresponding `.tex` file in the `report/chapters/` directory.
3. Once done, if you are completely confident, mark as `[x]`. If it needs human review or you are missing information, mark as `[NEEDS VERIFICATION]` and add a short comment below the task explaining what is needed.
4. Do not exceed the 15-page limit across the compiled document.

---

## Tasks

- [x] **1. Project Setup**
  - [x] Create report directory structure (`report/chapters/`).
  - [x] Scaffold `main.tex` and included files.

- [ ] **2. Abstract**
  - File: `report/chapters/00_abstract.tex`
  - Goal: Summarize the full multi-stage pipeline: data curation, classical CV audits, whole-image baseline, MIL, and OOD evaluation.

- [ ] **3. Introduction**
  - File: `report/chapters/01_introduction.tex`
  - [ ] Define the agricultural context and the problem of Flea Beetle damage in crops.
  - [ ] Introduce the full scope of the project: evaluating and comparing Classical CV heuristics, traditional Deep Learning (Whole-Image), and Weakly Supervised Learning (MIL).
  - [ ] Highlight the core contributions across ALL stages (curation, classical CV audits, deep learning pipelines, and rigorous OOD evaluation).
  - [ ] Outline the structure of the rest of the report.

- [ ] **4. Related Work / Background**
  - File: `report/chapters/02_related_work.tex`
  - [ ] Review traditional/classical Computer Vision techniques in agriculture.
  - [ ] Discuss standard deep learning approaches (Whole-image regression/classification).
  - [ ] Present Multiple Instance Learning (MIL) concepts.
  - [ ] Discuss Out-of-Distribution (OOD) generalization challenges.

- [ ] **5. Methodology**
  - File: `report/chapters/03_methodology.tex`
  - [ ] **5.1 Data Curation & Frame Extraction:** Detail how the raw data was processed.
  - [ ] **5.2 Classical Computer Vision Approaches:** Detail the classical CV methods used for plant proposal, frame auditing, green region extraction, and direct damage/hole detection.
  - [ ] **5.3 Whole-Image Baseline:** Describe the traditional deep learning architecture trained directly on whole images.
  - [ ] **5.4 Multiple Instance Learning (MIL):** Detail the feature extraction process and the Attention MIL architecture.
  - [ ] **5.5 Out-of-Distribution (OOD) Framework:** Explain the methodology for evaluating robustness on unseen datasets.

- [ ] **6. Experiments & Results**
  - File: `report/chapters/04_experiments.tex`
  - [ ] **6.1 Experimental Setup:** Describe dataset splits, hyperparameter configurations, and evaluation metrics.
  - [ ] **6.2 Classical CV Exploratory Audits:** Document and present the results from all heuristics and classical audits:
    - `frame_audit`, `plant_region_audit`, `plant_weight_audit`, `direct_damage_audit`, `hole_detection_comparison`, and `biology_audit`.
  - [ ] **6.3 Deep Learning Baselines:** Present the training and test results for the non-MIL deep learning baselines:
    - `baseline_regression_seed42`, `baseline_regression_mse_seed42`, and `patch_regression_seed42`.
  - [ ] **6.4 MIL Pipeline Results:** Analyze the results and provide a consolidated table for all MIL matrix experiments:
    - `mil_weighted_seed42`, `mil_abmil_seed42`, `patch_joint_seed42`, `patch_joint_sampled_seed42`, and the tests from `aggregation_experiments.json`.
  - [ ] **6.5 OOD Generalization:** Present the zero-shot quantitative results on all evaluated OOD datasets:
    - `Rauischholzhausen_WG1` and `DSV_Trial_1`.
  - [ ] **6.6 Qualitative Analysis:** Provide visual examples (attention maps, prediction vs true plots, classical CV segmentations) to interpret model decisions across all methods.

- [ ] **7. Conclusion & Future Work**
  - File: `report/chapters/05_conclusion.tex`
  - [ ] Summarize the comparative findings between Classical CV, Whole-Image, and MIL.
  - [ ] Discuss current limitations and edge cases observed across the different methods.
  - [ ] Propose future research directions.

- [ ] **8. Final Review**
  - [ ] Compile the LaTeX document and verify compilation succeeds.
  - [ ] Check the 15-page limit constraint.
  - [ ] Fix any formatting, typo, or citation issues.
