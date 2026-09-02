import pytest
import pandas as pd
import numpy as np
import cv2
import os
import torch
import tempfile
from PIL import Image

from src.data.patch_dataset import CSFBPlantPatchDataset, patch_collate_fn

@pytest.fixture
def dummy_manifest():
    # Create a temporary directory for dummy images
    temp_dir = tempfile.mkdtemp()
    
    # Create an image with some green spots
    img1_path = os.path.join(temp_dir, "img1.jpg")
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img1[10:30, 10:30] = [0, 255, 0] # one green square
    cv2.imwrite(img1_path, img1)
    
    # Create an image with NO green spots
    img2_path = os.path.join(temp_dir, "img2.jpg")
    img2 = np.zeros((100, 100, 3), dtype=np.uint8) # totally black
    cv2.imwrite(img2_path, img2)
    
    # Create a manifest
    manifest_path = os.path.join(temp_dir, "manifest.csv")
    df = pd.DataFrame({
        "image_path": [img1_path, img2_path],
        "mean_score": [12.5, 0.0],
        "plot_group": ["A1", "A2"],
        "split": ["train", "train"]
    })
    df.to_csv(manifest_path, index=False)
    
    return manifest_path, temp_dir

def test_extraction_and_shapes(dummy_manifest):
    manifest_path, _ = dummy_manifest
    
    # Use a small min_area so our 20x20 green square (area 400) is detected
    dataset = CSFBPlantPatchDataset(manifest_path, min_plant_area=100)
    
    patch_tensor, area_tensor, target, plot_group = dataset[0]
    
    # The first image has one green square
    assert patch_tensor.dim() == 4 # [N, C, H, W]
    assert patch_tensor.shape[0] == 1 # 1 patch
    assert patch_tensor.shape[1] == 3 # 3 channels
    # By default, without transform, ToTensor returns C,H,W based on the extracted patch
    # We padded by 15. The square is 20x20. With padding up to image edges (100x100)
    # The box is x:10, y:10, w:20, h:20. Padded box: 0 to 45 (45x45)
    assert patch_tensor.shape[2:] == (30, 30)
    
    assert area_tensor.shape == (1,)
    assert area_tensor[0].item() > 0
    assert target.item() == 12.5
    assert plot_group == "A1"

def test_empty_image_fallback(dummy_manifest):
    manifest_path, _ = dummy_manifest
    
    dataset = CSFBPlantPatchDataset(manifest_path, min_plant_area=100)
    
    # Second image is empty
    patch_tensor, area_tensor, target, plot_group = dataset[1]
    
    # Fallback should return a 224x224 black image tensor
    assert patch_tensor.dim() == 4
    assert patch_tensor.shape[0] == 1
    assert patch_tensor.shape[1] == 3
    assert patch_tensor.shape[2:] == (224, 224)
    
    assert area_tensor.shape == (1,)
    assert area_tensor[0].item() == 0.0
    assert target.item() == 0.0
    assert plot_group == "A2"

def test_patch_collate_fn():
    # Simulate a batch of 2 elements
    # element 1: 3 patches
    pt1 = torch.rand(3, 3, 224, 224)
    at1 = torch.tensor([100., 200., 300.])
    target1 = torch.tensor(10.0)
    pg1 = "G1"
    
    # element 2: 1 patch
    pt2 = torch.rand(1, 3, 224, 224)
    at2 = torch.tensor([150.])
    target2 = torch.tensor(20.0)
    pg2 = "G2"
    
    batch = [(pt1, at1, target1, pg1), (pt2, at2, target2, pg2)]
    
    patch_tensors, area_tensors, targets, plot_groups = patch_collate_fn(batch)
    
    assert isinstance(patch_tensors, list)
    assert len(patch_tensors) == 2
    assert patch_tensors[0].shape == (3, 3, 224, 224)
    assert patch_tensors[1].shape == (1, 3, 224, 224)
    
    assert isinstance(area_tensors, list)
    assert len(area_tensors) == 2
    assert area_tensors[0].shape == (3,)
    
    assert targets.dim() == 1
    assert targets.shape[0] == 2
    
    assert plot_groups == ["G1", "G2"]

def test_paired_dataset_and_collate_fn(dummy_manifest):
    manifest_path, _ = dummy_manifest
    
    from src.data.patch_dataset import CSFBPatchPairedDataset, patch_paired_collate_fn
    
    # We use margin=10. The dummy manifest has images with scores 12.5 and 0.0.
    # The difference is 12.5, which is > 10, so they should form 1 pair.
    dataset = CSFBPatchPairedDataset(manifest_path, min_plant_area=100, margin=10.0)
    
    # Dataset length is now equal to base dataset length
    assert len(dataset) == 2
    
    patch_A, area_A, target_A, group_A, patch_B, area_B, target_B, group_B = dataset[0]
    
    assert patch_A.dim() == 4
    assert patch_B.dim() == 4
    assert target_A.item() == 12.5
    assert target_B.item() == 0.0
    
    # Test collate function
    batch = [dataset[0]]
    
    (patch_tensors_A, area_tensors_A, targets_A, plot_groups_A, 
     patch_tensors_B, area_tensors_B, targets_B, plot_groups_B) = patch_paired_collate_fn(batch)
     
    assert len(patch_tensors_A) == 1
    assert len(patch_tensors_B) == 1
    assert targets_A.shape[0] == 1
    assert targets_B.shape[0] == 1
    assert targets_A[0].item() == 12.5
    assert targets_B[0].item() == 0.0
