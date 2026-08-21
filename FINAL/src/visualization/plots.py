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
