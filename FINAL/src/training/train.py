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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataset import get_dataloaders
from src.models.dinov3_regressor import DINOv3Regressor, get_loss_function
from src.models.losses import JointRankingRegressionLoss
from src.visualization.plots import plot_training_history

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(manifest, epochs=50, batch_size=32, lr=1e-3, loss="huber", patience=10, out_dir="outputs", seed=42, num_workers=4, image_size=224, training_mode="regression"):
    set_seed(seed)
    
    checkpoints_dir = os.path.join(out_dir, "checkpoints")
    logs_dir = os.path.join(out_dir, "logs")
    tb_dir = os.path.join(out_dir, "tensorboard")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Mode: {training_mode}")
    
    print(f"Loading data from {manifest}...")
    train_loader, val_loader, test_loader = get_dataloaders(
        manifest_path=manifest,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        high_quality_only=True,
        training_mode=training_mode
    )
    print(f"Dataloaders initialized. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    print("Initializing model...")
    model = DINOv3Regressor(head_width=256, dropout_p=0.3)
    model.to(device)
    
    head_params = [p for p in model.regression_head.parameters() if p.requires_grad]
    optimizer = AdamW(head_params, lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    if training_mode == 'joint':
        criterion = JointRankingRegressionLoss(margin=5.0, lambda_rank=0.5, delta=1.0)
    else:
        criterion = get_loss_function(loss_type=loss)
        
    mae_criterion = torch.nn.L1Loss()
    
    writer = SummaryWriter(tb_dir)
    csv_log_path = os.path.join(logs_dir, "training_log.csv")
    log_columns = ['epoch', 'train_loss', 'train_mae', 'val_loss', 'val_mae', 'lr', 'time_sec']
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
                img_A, img_B, tgt_A, tgt_B, _, _ = batch
                img_A, img_B = img_A.to(device), img_B.to(device)
                tgt_A, tgt_B = tgt_A.to(device), tgt_B.to(device)
                
                pred_A = model(img_A)
                pred_B = model(img_B)
                loss_val = criterion(pred_A, pred_B, tgt_A, tgt_B)
                
                loss_val.backward()
                optimizer.step()
                
                batch_size_real = tgt_A.size(0) * 2
                train_loss_sum += loss_val.item() * batch_size_real
                
                with torch.no_grad():
                    mae_sum = mae_criterion(pred_A, tgt_A).item() * tgt_A.size(0) + mae_criterion(pred_B, tgt_B).item() * tgt_B.size(0)
                    train_mae_sum += mae_sum
                    
            else:
                images, targets, _ = batch
                images, targets = images.to(device), targets.to(device)
                
                predictions = model(images)
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
        
        # Validation is always standard regression (MAE) since we want absolute performance
        # We don't evaluate ranking loss on val unless specifically required.
        val_criterion = get_loss_function(loss_type=loss)
        
        with torch.no_grad():
            for images, targets, _ in tqdm(val_loader, desc=f"Epoch {epoch:03d}/{epochs} [Val]"):
                images, targets = images.to(device), targets.to(device)
                
                predictions = model(images)
                loss_val = val_criterion(predictions, targets)
                
                val_loss_sum += loss_val.item() * targets.size(0)
                val_mae_sum += mae_criterion(predictions, targets).item() * targets.size(0)
                
        val_loss_epoch = val_loss_sum / len(val_loader.dataset)
        val_mae_epoch = val_mae_sum / len(val_loader.dataset)
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"Train Loss: {train_loss_epoch:.4f}, MAE: {train_mae_epoch:.4f} | "
              f"Val Loss: {val_loss_epoch:.4f}, MAE: {val_mae_epoch:.4f} | "
              f"Time: {epoch_time:.1f}s")
              
        writer.add_scalar('Loss/Train', train_loss_epoch, epoch)
        writer.add_scalar('MAE/Train', train_mae_epoch, epoch)
        writer.add_scalar('Loss/Val', val_loss_epoch, epoch)
        writer.add_scalar('MAE/Val', val_mae_epoch, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        log_data.append([epoch, train_loss_epoch, train_mae_epoch, val_loss_epoch, val_mae_epoch, current_lr, epoch_time])
        scheduler.step(val_mae_epoch)
        
        # --- EARLY STOPPING & CHECKPOINTING ---
        if val_mae_epoch < best_val_mae:
            best_val_mae = val_mae_epoch
            patience_counter = 0
            best_model_path = os.path.join(checkpoints_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mae': best_val_mae
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
    
    # Generate training plots automatically
    plots_dir = os.path.join(out_dir, "plots")
    plot_training_history(csv_log_path, plots_dir)
    
    print("Training complete.")
