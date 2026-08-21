import os
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

def evaluate_model(manifest, model_path, batch_size=32, out_dir="outputs", num_workers=4, image_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
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
    model = DINOv3Regressor(head_width=256, dropout_p=0.0) # Dropout 0 for eval, though eval() disables it anyway
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
        
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # Support loading raw state dicts too
        
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
    
    return results_df
