import os
import time
import random
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.patch_dataset import get_patch_dataloaders
from src.models.patch_model import DINOv3PatchRegressor
from src.models.dinov3_regressor import get_loss_function
from src.models.losses import JointRankingRegressionLoss
from src.visualization.plots import plot_training_history
from src.training.provenance import build_run_metadata, save_json
from scipy.stats import spearmanr, pearsonr

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_constants(train_loader, val_loader, device):
    """ Evaluate constant mean and median predictors on validation. """
    if hasattr(train_loader.dataset, 'base_dataset'):
        train_targets = train_loader.dataset.base_dataset.df['mean_score'].values
    else:
        train_targets = train_loader.dataset.df['mean_score'].values
        
    if hasattr(val_loader.dataset, 'base_dataset'):
        val_targets = val_loader.dataset.base_dataset.df['mean_score'].values
    else:
        val_targets = val_loader.dataset.df['mean_score'].values
    
    train_mean = np.mean(train_targets)
    train_median = np.median(train_targets)
    
    mean_mae = np.mean(np.abs(val_targets - train_mean))
    median_mae = np.mean(np.abs(val_targets - train_median))
    
    return {
        "train_mean": float(train_mean),
        "mean_mae": float(mean_mae),
        "train_median": float(train_median),
        "median_mae": float(median_mae)
    }

def train_patch_model(manifest, epochs=50, batch_size=32, lr=1e-3, loss="huber", patience=10,
                      out_dir="outputs/runs", run_name="baseline_patch_seed42", seed=42, num_workers=4,
                      image_size=224, high_quality_only=False,
                      model_name="dinov3_vits16", head_width=256,
                      dropout_p=0.3, weights_path=None, aggregation="weighted",
                      training_mode="regression", joint_margin=5.0):
    set_seed(seed)

    run_dir = os.path.join(out_dir, run_name)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")
    tb_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Aggregation: {aggregation} | Run: {run_name}")

    model_config = {
        "model_name": model_name,
        "head_width": head_width,
        "dropout_p": dropout_p,
        "aggregation": aggregation,
        "local_weights_path": weights_path,
    }
    training_config = {
        "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
        "loss": loss, "patience": patience, "seed": seed,
        "num_workers": num_workers, "high_quality_only": high_quality_only,
    }
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    run_metadata = build_run_metadata(
        project_dir, manifest, run_name, model_config, training_config
    )
    if weights_path:
        from src.training.provenance import sha256_file
        weights_file = os.path.join(weights_path, "model.safetensors")
        run_metadata["weights_sha256"] = sha256_file(weights_file)
    save_json(run_metadata, os.path.join(run_dir, "run_config.json"))
    
    print(f"Loading data from {manifest}...")
    train_loader, val_loader, test_loader = get_patch_dataloaders(
        manifest_path=manifest,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        high_quality_only=high_quality_only,
        training_mode=training_mode,
        joint_margin=joint_margin
    )
    print(f"Dataloaders initialized. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Compute Constant Baselines
    constants = evaluate_constants(train_loader, val_loader, device)
    print(f"Constant Baseline - Train Mean: {constants['train_mean']:.4f} -> Val MAE: {constants['mean_mae']:.4f}")
    print(f"Constant Baseline - Train Median: {constants['train_median']:.4f} -> Val MAE: {constants['median_mae']:.4f}")
    
    print("Initializing Patch Aggregator Model...")
    model = DINOv3PatchRegressor(**model_config)
    model.to(device)
    
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = AdamW(head_params, lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    if training_mode == 'joint':
        criterion = JointRankingRegressionLoss(margin=joint_margin)
    else:
        criterion = get_loss_function(loss_type=loss)
    mae_criterion = torch.nn.L1Loss()
    
    writer = SummaryWriter(tb_dir)
    csv_log_path = os.path.join(logs_dir, "training_log.csv")
    log_columns = ['epoch', 'train_loss', 'train_mae', 'val_loss', 'val_mae', 'val_spearman', 'lr', 'time_sec']
    log_data = []
    
    best_val_mae = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # --- TRAIN PHASE ---
        model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs} [Train]"):
            optimizer.zero_grad()
            
            if training_mode == 'joint':
                patch_tensors_A, area_tensors_A, targets_A, _, patch_tensors_B, area_tensors_B, targets_B, _ = batch
                
                patch_tensors_A = [p.to(device) for p in patch_tensors_A]
                area_tensors_A = [a.to(device) for a in area_tensors_A]
                targets_A = targets_A.to(device).unsqueeze(1)
                
                patch_tensors_B = [p.to(device) for p in patch_tensors_B]
                area_tensors_B = [a.to(device) for a in area_tensors_B]
                targets_B = targets_B.to(device).unsqueeze(1)
                
                predictions_A = model(patch_tensors_A, area_tensors_A)
                predictions_B = model(patch_tensors_B, area_tensors_B)
                
                loss_val = criterion(predictions_A, predictions_B, targets_A, targets_B)
                
                loss_val.backward()
                optimizer.step()
                
                batch_size_real = targets_A.size(0) * 2
                train_loss_sum += loss_val.item() * batch_size_real
                train_mae_sum += (mae_criterion(predictions_A, targets_A).item() * targets_A.size(0) + 
                                  mae_criterion(predictions_B, targets_B).item() * targets_B.size(0))
            else:
                patch_tensors, area_tensors, targets, _ = batch
                patch_tensors = [p.to(device) for p in patch_tensors]
                area_tensors = [a.to(device) for a in area_tensors]
                targets = targets.to(device).unsqueeze(1)
                
                predictions = model(patch_tensors, area_tensors)
                loss_val = criterion(predictions, targets)
                
                loss_val.backward()
                optimizer.step()
                
                batch_size_real = targets.size(0)
                train_loss_sum += loss_val.item() * batch_size_real
                train_mae_sum += mae_criterion(predictions, targets).item() * batch_size_real
            
        train_loss_epoch = train_loss_sum / (len(train_loader.dataset) * (2 if training_mode == 'joint' else 1))
        train_mae_epoch = train_mae_sum / (len(train_loader.dataset) * (2 if training_mode == 'joint' else 1))
        
        # --- VAL PHASE ---
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        
        val_criterion = get_loss_function(loss_type=loss)
        
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for patch_tensors, area_tensors, targets, _ in tqdm(val_loader, desc=f"Epoch {epoch:03d}/{epochs} [Val]"):
                patch_tensors = [p.to(device) for p in patch_tensors]
                area_tensors = [a.to(device) for a in area_tensors]
                targets = targets.to(device).unsqueeze(1)
                
                predictions = model(patch_tensors, area_tensors)
                loss_val = val_criterion(predictions, targets)
                
                val_loss_sum += loss_val.item() * targets.size(0)
                val_mae_sum += mae_criterion(predictions, targets).item() * targets.size(0)
                
                all_val_preds.extend(predictions.cpu().squeeze().numpy())
                all_val_targets.extend(targets.cpu().squeeze().numpy())
                
        val_loss_epoch = val_loss_sum / len(val_loader.dataset)
        val_mae_epoch = val_mae_sum / len(val_loader.dataset)
        
        # Compute correlations
        if len(all_val_preds) > 1:
            spearman_rho, _ = spearmanr(all_val_targets, all_val_preds)
        else:
            spearman_rho = 0.0
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"Train Loss: {train_loss_epoch:.4f}, MAE: {train_mae_epoch:.4f} | "
              f"Val Loss: {val_loss_epoch:.4f}, MAE: {val_mae_epoch:.4f}, Rho: {spearman_rho:.4f} | "
              f"Time: {epoch_time:.1f}s")
              
        writer.add_scalar('Loss/Train', train_loss_epoch, epoch)
        writer.add_scalar('MAE/Train', train_mae_epoch, epoch)
        writer.add_scalar('Loss/Val', val_loss_epoch, epoch)
        writer.add_scalar('MAE/Val', val_mae_epoch, epoch)
        writer.add_scalar('Spearman/Val', spearman_rho, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        log_data.append([epoch, train_loss_epoch, train_mae_epoch, val_loss_epoch, val_mae_epoch, spearman_rho, current_lr, epoch_time])
        scheduler.step(val_mae_epoch)
        
        # --- EARLY STOPPING & CHECKPOINTING ---
        if val_mae_epoch < best_val_mae:
            best_val_mae = val_mae_epoch
            patience_counter = 0
            best_model_path = os.path.join(checkpoints_dir, "best_model.pth")
            torch.save({
                'checkpoint_schema_version': 1,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mae': best_val_mae,
                'run_metadata': run_metadata,
                'model_config': model_config,
                'training_config': training_config,
                'constants': constants
            }, best_model_path)
            print(f"   -> New best validation MAE: {best_val_mae:.4f}! Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    df_log = pd.DataFrame(log_data, columns=log_columns)
    df_log.to_csv(csv_log_path, index=False)
    print(f"Training log saved to {csv_log_path}")
    writer.close()
    
    plots_dir = os.path.join(run_dir, "plots")
    plot_training_history(csv_log_path, plots_dir)
    
    print("Training complete.")
