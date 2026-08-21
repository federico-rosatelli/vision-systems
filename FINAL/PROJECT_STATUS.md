# Project Status: CSFB Damage Quantification

## Completed Phases (Infrastructure & Development)

- **Phase 0: Data Preparation & Cleaning**
  - [x] Script to merge scores from scattered CSVs (`make_manifest.py`).
  - [x] Human disagreement handling and filtering.
  - [x] Isolation of the 470 "high-quality" images (baseline).

- **Phase 1: Evaluation Protocol**
  - [x] Stratified split into Train (70%), Val (10%), Test (20%).
  - [x] Data leakage protection by grouping at the `plot_group` level.
  - [x] PyTorch DataLoader for DINOv3 with data augmentation (`dataset.py`).

- **Phase 2: Baseline Model & Training Loop**
  - [x] Integration of `DINOv3` backbone (frozen) + `MLP Regression Head` (clamped to 0-100%).
  - [x] Central orchestrator `main.py` to manage the entire pipeline.
  - [x] Integration of TensorBoard, Checkpointing, Huber Loss, and Early Stopping.
  - [x] Automated evaluation script with metric extraction (MAE, RMSE, Pearson, Spearman) and visual plots.

- **Phase 3: Ranking-Based Weak Supervision**
  - [x] Creation of paired dataset (`CSFBPairedDataset`).
  - [x] Development of hybrid cost function: Huber + MarginRankingLoss (`JointRankingRegressionLoss`).
  - [x] Support for joint training in `main.py` via the `--training_mode joint` flag.

- **Phase 5: Genotype Evaluation**
  - [x] Prediction aggregation algorithm grouped by `plot_group` / `genotype` (`genotype_ranking.py`).

- **Phase 6: Inference Tools**
  - [x] Command-line script for fast inference on single images (`predict.py`).

- **Automation & Testing**
  - [x] Unit tests and end-to-end tests implemented and passed.

---

## Pending Phases (Execution & Research)

- **Official Training Execution**
  - [ ] Run standard training (Absolute Regression) on the real dataset.
  - [ ] Run advanced training (Pairwise Ranking) on the real dataset.
  - [ ] Compare test metrics between the two models to select the best one.

- **Biological Results Extraction**
  - [ ] Run ranking analysis on real data to produce the final leaderboard of resistant genotypes to hand over to the team.

---

## Optional Developments / Stretch Goals

- **Phase 4: Agronomic Features Extraction**
  - [ ] (Optional) Integrate YOLO or Segment Anything (SAM) to mathematically count holes instead of relying on direct regression of the whole leaf.

- **User Interface (Web-App)**
  - [ ] (Optional) Develop an interactive app in Streamlit/Gradio to simplify usage for agronomists.

---

## Currently Available Commands
The entire pipeline is managed via the `main.py` file:

```bash
# Prepare CSVs and images
python main.py prepare_data --raw_dir /path/to/data

# Start standard training
python main.py train --training_mode regression

# Start advanced training (pairs)
python main.py train --training_mode joint

# Evaluate performance on the test set
python main.py evaluate --model_path outputs/checkpoints/best_model.pth

# Plot the training history logs (Loss, MAE, LR)
python main.py plot_logs --log_file outputs/logs/training_log.csv --out_dir outputs/plots

# Get the genotype resistance leaderboard
python main.py rank --preds_file outputs/tables/test_predictions.csv

# Analyze a single leaf
python main.py predict --image my_leaf.jpg
```
