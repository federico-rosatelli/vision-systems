import os
import sys
import torch
import pandas as pd
from PIL import Image
import numpy as np

# Add the src directory to the python path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.make_manifest import build_manifest
from src.data.make_splits import create_splits
from src.data.dataset import get_dataloaders
from src.models.dinov3_regressor import DINOv3Regressor

def setup_mock_data(base_dir):
    """Generates synthetic images and a CSV to simulate the raw data."""
    os.makedirs(base_dir, exist_ok=True)
    
    data = []
    # Create 30 mock images (10 plots, 3 images each)
    for i in range(30):
        plot_id = f"plot_{i // 3}"
        filename = f"2025_10_14_BBCH10_{i}.jpg"
        
        # Simulate some JLU and GAU scores
        score_jlu = np.random.uniform(5, 40)
        # Force some to be high-quality (diff < 5) and some to be low-quality
        noise = np.random.uniform(-2, 2) if i % 2 == 0 else np.random.uniform(-10, 10)
        score_gau = score_jlu + noise 
        
        data.append({
            'filename': filename,
            'Score_JLU': score_jlu,
            'Score_GAU': score_gau,
            'plot_id': plot_id
        })
        
        # Create a small valid JPEG mock image
        img = Image.new('RGB', (224, 224), color=(np.random.randint(0,255), 100, 50))
        img.save(os.path.join(base_dir, filename))
        
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(base_dir, 'mock_raw_scores.csv'), index=False)
    print(f"Mock data generated in {base_dir}")

def test_full_pipeline():
    print("--- STARTING TESTS ---")
    mock_dir = os.path.join(os.path.dirname(__file__), "mock_data")
    out_dir = os.path.join(os.path.dirname(__file__), "mock_outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    out_manifest = os.path.join(out_dir, "data_manifest.csv")
    out_splits = os.path.join(out_dir, "data_manifest_split.csv")
    
    # 1. Setup Data
    print("\n--- Testing Data Setup ---")
    setup_mock_data(mock_dir)
    
    # 2. Make Manifest
    print("\n--- Testing build_manifest ---")
    build_manifest(mock_dir, out_manifest)
    assert os.path.exists(out_manifest), "Manifest file was not created!"
    
    # 3. Make Splits
    print("\n--- Testing create_splits ---")
    create_splits(out_manifest, out_splits, random_state=42)
    assert os.path.exists(out_splits), "Splits file was not created!"
    
    # Verify no plot_id leaked across splits
    split_df = pd.read_csv(out_splits)
    train_plots = set(split_df[split_df['split'] == 'train']['plot_group'])
    val_plots = set(split_df[split_df['split'] == 'val']['plot_group'])
    test_plots = set(split_df[split_df['split'] == 'test']['plot_group'])
    
    assert train_plots.isdisjoint(val_plots), "Leakage detected between train and val!"
    assert train_plots.isdisjoint(test_plots), "Leakage detected between train and test!"
    assert val_plots.isdisjoint(test_plots), "Leakage detected between val and test!"
    print("Leakage test passed. No plot leaks across splits.")
    
    # 4. DataLoaders
    print("\n--- Testing DataLoaders ---")
    # Setting high_quality_only to False so we have enough data for the batch
    train_loader, val_loader, test_loader = get_dataloaders(
        out_splits, batch_size=2, num_workers=0, image_size=224, high_quality_only=False
    )
    
    batch_img, batch_tgt, batch_plot = next(iter(train_loader))
    assert batch_img.shape == (2, 3, 224, 224), f"Wrong image shape: {batch_img.shape}"
    assert batch_tgt.shape == (2,), f"Wrong target shape: {batch_tgt.shape}"
    print("Dataloader yields correct tensor shapes.")
    
    # 5. Model Forward Pass
    print("\n--- Testing Model Forward Pass ---")
    model = DINOv3Regressor(
        model_name='dinov3_vits16', weights_path='weights/dinov3-vits16-hf',
        head_width=32, image_size=224
    )
    model.eval() # Ensure eval mode
    
    # Since backbone is frozen, gradients should only be required for the regression_head
    for name, param in model.named_parameters():
        if 'backbone' in name:
            assert not param.requires_grad, f"Backbone parameter {name} should be frozen!"
        if 'regression_head' in name:
            assert param.requires_grad, f"Head parameter {name} should require gradients!"
    print("Gradient requirements test passed.")
            
    out = model(batch_img)
    assert out.shape == (2,), f"Wrong output shape: {out.shape}"
    assert torch.all((out >= 0) & (out <= 100)), "Outputs are out of bounds [0, 100]"
    print("Model forward pass successful. Shape and bounds are correct.")
    
    print("\n✅ ALL TESTS PASSED PERFECTLY!")
