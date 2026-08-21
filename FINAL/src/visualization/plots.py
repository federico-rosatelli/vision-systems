import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_pred_vs_true(y_true, y_pred, save_path):
    """Generates a scatter plot of Predictions vs Ground Truth."""
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color='blue', edgecolor='w')
    
    # Perfect prediction line
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.title('Predicted vs Actual CSFB Damage (%)')
    plt.xlabel('Ground Truth (%)')
    plt.ylabel('Predicted (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_residuals(y_true, y_pred, save_path):
    """Generates a histogram and scatter plot of residuals."""
    residuals = np.array(y_pred) - np.array(y_true)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=ax1, color='purple', bins=30)
    ax1.set_title('Residuals Distribution')
    ax1.set_xlabel('Prediction Error (%)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(x=0, color='r', linestyle='--', lw=2)
    
    # Residuals vs True Values
    sns.scatterplot(x=y_true, y=residuals, alpha=0.6, ax=ax2, color='teal')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_title('Residuals vs Ground Truth')
    ax2.set_xlabel('Ground Truth (%)')
    ax2.set_ylabel('Residuals (%)')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_training_history(csv_path, out_dir):
    """
    Reads the training_log.csv and generates various plots for training history.
    """
    import pandas as pd
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Loss Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='epoch', y='train_loss', label='Train Loss', marker='o')
    sns.lineplot(data=df, x='epoch', y='val_loss', label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. MAE Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='epoch', y='train_mae', label='Train MAE', marker='s', color='green')
    sns.lineplot(data=df, x='epoch', y='val_mae', label='Validation MAE', marker='s', color='red')
    plt.title('Mean Absolute Error (MAE) over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(out_dir, "mae_curve.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Learning Rate Plot
    if 'lr' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='epoch', y='lr', color='orange', marker='^')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(out_dir, "lr_curve.png"), bbox_inches='tight', dpi=300)
        plt.close()
        
    print(f"Training plots saved to {out_dir}")
