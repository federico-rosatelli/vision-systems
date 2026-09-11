# Final Project Results: Cabbage Stem Flea Beetle (CSFB) Leaf Damage Quantification

This document compiles the complete benchmark results, empirical tables, model selection audits, and preprocessing evaluations for the Cabbage Stem Flea Beetle (CSFB) plant damage quantification project.

---

## 1. Executive Summary & Objective

The primary objective is to build a **reproducible computer vision pipeline** that quantifies CSFB feeding damage on oilseed rape (*Brassica napus*) plants within a standardized reference frame. The target metrics are the percentage of visible leaf area affected by **shot holes** and **yellow/brown pitting** for BBCH10 growth stage field images, supporting downstream genotype resistance analysis.

---

## 2. Data Contract & Fixed Dataset Splits

All in-domain development, model selection, and validation strictly adhere to the leakage-safe manifest `outputs/tables/baseline_manifest_split.csv`.

| Dataset Attribute | Value | Verification & Integrity Safeguards |
| :--- | :---: | :--- |
| **Total Verified Population (BBCH10)** | **470 images** | Unique physical images with consensus JLU/GAU expert scores |
| **Training Split (`train`)** | **331 images** | Zero plot-group (`plot_group`) overlap with val/test |
| **Validation Split (`val`)** | **66 images** | Used exclusively for model selection and early stopping |
| **Frozen Test Split (`test`)** | **73 images** | **Kept strictly closed** during model optimization |
| **Cache Manifest Hash (SHA-256)** | Verified | Recorded in `outputs/cache/dinov3_bags.pt` |

---

## 3. Preprocessing & Crop Pipeline Audits

Before modeling, the preprocessing pipeline (metal reference frame detection and plant region proposal) was manually audited on a stratified sample of 30 train/validation images.

| Preprocessing Module | Sample Size | Automatic Success | Manual Approval Rate | Key Audit Metrics |
| :--- | :---: | :---: | :---: | :--- |
| **Metal Reference Frame Detection** | 30 images | 30/30 (100%) | **100.0% (30/30)** | Mean frame confidence: 0.8123 (Reviewed by Emrullah) |
| **Plant Region Proposal Audit** | 30 images | 30/30 (100%) | **100.0% (30/30)** | Extracted 191 plant regions (Median: 6.0 plants/image) |

---

## 4. Whole-Image DINOv3 Baseline Performance

Evaluation of the whole-image baseline model, where complete field frames are downsampled directly to $224 \times 224$ pixels without high-resolution patch extraction.

| Evaluation Metric | Validation Set ($N=66$) | Frozen Test Set ($N=73$) |
| :--- | :---: | :---: |
| **MAE (Mean Absolute Error %)** | 5.1265 | 4.6502 |
| **Pearson Correlation ($r$)** | 0.2391 | -0.0971 |
| **Spearman Rank Correlation ($\rho$)** | 0.2712 | -0.1078 |

> **Key Finding**: Downsampling whole images discards small shot-hole details, leading to negative test correlations. This strongly justified moving to a Multiple Instance Learning (MIL) patch-based architecture.

---

## 5. Controlled MIL Aggregation Experiment (3-Seed Audit Across 12 Runs)

Across-seed validation performance (seeds 42, 43, 44) comparing plant-score aggregation functions on `baseline_manifest_split.csv`.

| MIL Aggregation Function | Validation MAE (Mean ± SD) | Spearman Rank $\rho$ (Mean ± SD) |
| :--- | :---: | :---: |
| **Area-Weighted Plant Scores** ($w_i = A_i / \sum A_j$) | **2.6932 ± 0.0232** | **0.8532 ± 0.0022** |
| **Uniform Plant Scores** (Unweighted Mean) | 2.7293 ± 0.0401 | 0.8381 ± 0.0064 |
| **ABMIL** (Attention MIL - Ilse et al., 2018) | 2.7430 ± 0.0307 | 0.8093 ± 0.0109 |
| **Gated ABMIL** (Gated Attention MIL) | 2.7808 ± 0.0243 | 0.8046 ± 0.0107 |

---

## 6. Selected Production Model Performance (`mil_weighted_seed42`)

Detailed metrics for the selected Area-Weighted MIL model checkpoint on the validation set.

| Model Metric | Performance | Reference Baseline / Target |
| :--- | :---: | :---: |
| **Validation MAE (%)** | **2.6670** | Constant Mean MAE: 5.45% |
| **Validation RMSE (%)** | **3.5830** | Constant Mean RMSE: 6.85% |
| **Pearson Correlation ($r$)** | **0.8587** | - |
| **Spearman Rank Correlation ($\rho$)** | **0.8553** | - |
| **Pairwise Ranking Accuracy (Gap ≥ 5%)** | **0.9400 (94.0%)** | Pairwise severity ranking |

---

## 7. Zero-Shot Out-of-Distribution (OOD) Field Benchmark

Zero-shot evaluation of `mil_weighted_seed42` on unseen field trial datasets.

| OOD Benchmark Dataset | Folder Identifier | Valid Samples ($N$) | MAE (%) | RMSE (%) | Pearson $r$ | Spearman $\rho$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Rauischholzhausen WG1** | `2025_10_07_RSFB-Phenotyping_WG1_JLU` | **906** | **4.29%** | **5.83%** | **0.293** | **0.199** |
| **DSV Trial 1** | `2025_09_15_Res4StRes_T1_DSV` | **897** | **15.60%** | **17.24%** | **0.207** | **0.193** |

---

## 8. Classical Computer Vision Audits (Exploratory Work)

Audits evaluating classical computer vision algorithms as exploratory, explainable visual tools complementary to neural MIL regression.

| Classical CV Audit Module | Sample Size | Manual Usability Rate | Acceptance Status | Key Review Findings |
| :--- | :---: | :---: | :---: | :--- |
| **Plant Mask & Area Weight Audit** | 40 images | **100.0% (40/40)** | **Approved** | Plant masks accurately approximate visible leaf area for MIL weights. |
| **Direct Damage Audit** (Holes + Pitting) | 40 images | **0.0% (0/40)** | *Rejected for Holes* | Leaf masks & pitting are acceptable; automated shot-hole detection yields false positives. |
| **Soil-Aware Hole Detection Comparison** | 20 images | **0.0% (0/20)** | *Rejected* | Soil similarity thresholds confuse dark brown pitting with exposed soil below. |

---

## 9. Complete Project JSON Configuration Suite

The project configurations are organized into structured JSON files located in `configs/`:

### A. Core MIL Neural Pipeline Workflows
| Configuration File | Main Action (`action`) | Purpose & Description |
| :--- | :--- | :--- |
| **`configs/config_mil_data.json`** | `cache_embeddings` | Data preparation, split verification, and DINOv3 feature bag extraction (`outputs/cache/dinov3_bags.pt`). |
| **`configs/config_mil_train.json`** | `train_mil` | Area-Weighted MIL model training (`mil_weighted_seed42`), supporting `weighted`, `abmil`, and `gated_abmil`. |
| **`configs/config_mil_eval.json`** | `evaluate_ood` / `evaluate_mil` | In-distribution test set evaluation and zero-shot OOD field trial benchmarking. |
| **`configs/config.json`** | Master Default | Master fallback configuration used when `python main.py` is launched without parameters. |

### B. Baselines, Matrix Experiments & Pipeline Testing
| Configuration File | Main Action (`action`) | Purpose & Description |
| :--- | :--- | :--- |
| **`configs/config_mse.json`** | `train` | Whole-image baseline regression training using MSE loss. |
| **`configs/aggregation_experiments.json`** | `run_matrix` | 12-run grid experiment across 4 aggregation types (`weighted`, `uniform`, `abmil`, `gated_abmil`) and 3 seeds. |
| **`configs/smoke_config.json`** | Fast Smoke Test | Lightweight single-epoch configuration for rapid end-to-end pipeline verification. |

### C. Preprocessing & Classical Computer Vision Audits
| Configuration File | Audit Module | Purpose & Description |
| :--- | :--- | :--- |
| **`configs/frame_audit.json`** | Reference Frame Audit | Audit of metal reference frame cropping across 30 sample images. |
| **`configs/plant_region_audit.json`** | Plant Proposal Audit | Audit of plant region bounding box proposal across 30 sample images. |
| **`configs/plant_weight_audit.json`** | Area Weight Audit | Audit of plant mask pixel area weights across 40 sample images. |
| **`configs/direct_damage_audit.json`** | Direct Damage Audit | Audit of direct CV leaf area, hole, and pitting measurement across 40 images. |
| **`configs/hole_detection_comparison.json`** | Hole Classification Comparison | Comparison between HSV brightness vs CIELAB soil-aware hole detection across 20 difficult images. |
| **`configs/biology_audit.json`** | Biological Feature Audit | Audit of morphological biological feature extraction. |

---

## 10. Automated Verification & Reproducibility

- **Test Suite (`pytest`)**: **44 passed, 4 skipped in 28.96s**.
