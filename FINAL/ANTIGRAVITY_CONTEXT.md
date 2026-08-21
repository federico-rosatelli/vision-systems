# CSFB Feeding-Damage Quantification - Antigravity Context

## 1. Project Overview
This project aims to build a machine learning pipeline to quantify the average feeding damage (percentage) caused by the cabbage stem flea beetle (CSFB) on rapeseed plants. The ultimate biological goal is to rank different plant genotypes to identify which ones are naturally resistant to the beetle.

**Key Challenges:**
- Subjective human labeling (r=0.549 correlation between human experts).
- Limited high-quality labeled data (only 470 images).
- High risk of data leakage because there are 3 pictures per field plot (they are highly correlated).

## 2. Current State of the Repository
The foundational infrastructure (Phase 0, Phase 1, and the model architecture of Phase 2) has been fully implemented and tested.

Current File Structure:
- `src/data/make_manifest.py`: Consolidates raw CSV files, handles inconsistent formats, and extracts the 470 "High Quality" images (disagreement < 5%).
- `src/data/make_splits.py`: Creates strict group-aware, score-stratified Train/Val/Test splits (70/15/15). Leakage-proof by keeping `plot_group` intact across splits.
- `src/data/dataset.py`: PyTorch Dataset implementation. Includes DINO normalization and safe data augmentation (avoiding aggressive crops).
- `src/models/dinov3_regressor.py`: PyTorch Module. Loads a frozen DINO backbone (uses DINOv2 as placeholder if DINOv3 is unavailable) and adds a tunable MLP regression head that outputs a score strictly bounded to [0, 100].
- `tests/test_pipeline.py`: A complete end-to-end test verifying everything from mock data creation to the model forward pass. (ALL TESTS PASSED).

## 3. Strict Rules for Antigravity
When generating new code or modifying existing files for this project, you MUST strictly adhere to the following rules:
1. **Language:** All code, docstrings, comments, and console print outputs MUST be written in pure English.
2. **Encoding:** Use ONLY pure ASCII characters. Absolutely NO emojis or special symbols anywhere.
3. **Leakage:** Never evaluate single images independently across splits. Images belonging to the same plot/QR code must remain aggregated and evaluated together.
4. **Data Contract:** The model output must always represent a percentage bounded between 0 and 100.

## 4. Immediate Next Steps
The project must now continue following these phases:

### Phase 2 Continuation (Immediate Task)
- Implement `src/training/train.py`.
- This script needs to load the dataloaders, initialize `DINOv3Regressor`, and train the MLP head using MSE or Huber Loss.
- Implement Early Stopping based on Validation MAE (Mean Absolute Error).
- Integrate a logger (TensorBoard or standard CSV logging).

### Phase 3 - Ranking-Based Weak Supervision
- Modify the training approach to rely on relative comparisons rather than absolute labels (which are noisy).
- Implement a pairwise ranking loss (e.g., Bradley-Terry or Pairwise BCE) where the model learns that "Image A > Image B" if the manual label gap is > 5%.

### Phase 4 - Agronomic Features (Stretch Goal)
- Create `src/models/segmentation.py`.
- Integrate SAM 3 (Segment Anything 3) or YOLO/RF-DETR to isolate single plants.
- Add features to count shot-holes, count yellow/brown spots (pitting), and calculate average damage per plant.

### Phase 5 - Genotype Evaluation
- Aggregate image scores -> plot scores -> genotype scores.
- Compare AI rankings against manual human rankings to validate biological resistance.
