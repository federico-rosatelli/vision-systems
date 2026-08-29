import os
import json
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataset import get_dataloaders
from src.models.dinov3_regressor import DINOv3Regressor
from src.visualization.plots import plot_pred_vs_true, plot_residuals


def pairwise_ranking_accuracy(y_true, y_pred, minimum_gap):
    correct = 0
    total = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            true_difference = y_true[i] - y_true[j]
            if abs(true_difference) < minimum_gap:
                continue
            predicted_difference = y_pred[i] - y_pred[j]
            correct += (true_difference > 0) == (predicted_difference > 0)
            total += 1
    return {"accuracy": correct / total if total else None, "pair_count": total}

def evaluate_model(manifest, model_path, batch_size=32, out_dir=None, num_workers=4, image_size=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('model_config')
    if not model_config:
        raise ValueError("Checkpoint lacks required model_config metadata; retrain it as a reproducible run")
    checkpoint_image_size = model_config['image_size']
    if image_size is not None and image_size != checkpoint_image_size:
        raise ValueError("Requested image size does not match the checkpoint metadata")
    image_size = checkpoint_image_size
    if out_dir is None:
        out_dir = str(os.path.dirname(os.path.dirname(model_path)))

    # 1. Load Test Dataloader
    print(f"Loading test split from {manifest}...")
    _, _, test_loader = get_dataloaders(
        manifest_path=manifest,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        high_quality_only=True
    )
    
    if len(test_loader) == 0:
        print("Warning: Test loader is empty. Make sure your manifest has a 'test' split with valid scores.")
        return
        
    # 2. Load Model
    print(f"Loading model from {model_path}...")
    model = DINOv3Regressor(**model_config)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("Checkpoint lacks model_state_dict")
        
    model.to(device)
    model.eval()
    
    # 3. Inference Loop
    y_true = []
    y_pred = []
    plot_groups = []
    
    print("Running inference on the test set...")
    with torch.no_grad():
        for images, targets, groups in test_loader:
            images = images.to(device)
            preds = model(images).cpu().numpy()
            
            y_pred.extend(preds)
            y_true.extend(targets.numpy())
            plot_groups.extend(groups)
            
    # 4. Compute Metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Handle cases where all predictions or targets are the same (e.g. dummy test data)
    try:
        pearson_r, _ = pearsonr(y_true, y_pred)
    except Exception:
        pearson_r = 0.0
        
    try:
        spearman_rho, _ = spearmanr(y_true, y_pred)
    except Exception:
        spearman_rho = 0.0
        
    print("\n=== Evaluation Results ===")
    print(f"Total Test Samples : {len(y_true)}")
    print(f"MAE                : {mae:.4f} %")
    print(f"RMSE               : {rmse:.4f} %")
    print(f"Pearson r          : {pearson_r:.4f}")
    print(f"Spearman rho       : {spearman_rho:.4f}")
    print("==========================\n")
    
    # 5. Save Results & Plots
    plots_dir = os.path.join(out_dir, "plots")
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    plot_pred_vs_true(y_true, y_pred, os.path.join(plots_dir, "test_pred_vs_true.png"))
    plot_residuals(y_true, y_pred, os.path.join(plots_dir, "test_residuals.png"))
    print(f"Saved evaluation plots to {plots_dir}")
    
    results_df = pd.DataFrame({
        'plot_group': plot_groups,
        'true_score': y_true,
        'pred_score': y_pred,
        'absolute_error': np.abs(y_true - y_pred)
    })
    
    results_csv = os.path.join(tables_dir, "test_predictions.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Saved raw predictions to {results_csv}")

    metrics = {
        "test_samples": len(y_true),
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
        "pairwise_ranking": {
            str(gap): pairwise_ranking_accuracy(y_true, y_pred, gap)
            for gap in (5.0, 10.0, 20.0)
        },
        "checkpoint_path": os.path.abspath(model_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_mae": checkpoint.get("best_val_mae"),
        "run_metadata": checkpoint.get("run_metadata"),
    }
    metrics_path = os.path.join(tables_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved metrics to {metrics_path}")
    
    return results_df
