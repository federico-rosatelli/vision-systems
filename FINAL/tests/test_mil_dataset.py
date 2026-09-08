import os
import pytest
import torch
import pandas as pd
from src.data.patch_dataset import (
    CSFBCachedBagDataset,
    CSFBCachedPairedBagDataset,
    get_patch_dataloaders
)

@pytest.fixture
def mock_cache_file(tmp_path):
    cache_file = tmp_path / "test_dinov3_bags.pt"
    bags = [
        {
            'idx': 0,
            'filename': 'img1.jpg',
            'features': torch.randn(4, 384),
            'areas': torch.tensor([100.0, 150.0, 200.0, 250.0]),
            'target': 12.5,
            'plot_group': 'plot_A',
            'split': 'train'
        },
        {
            'idx': 1,
            'filename': 'img2.jpg',
            'features': torch.randn(6, 384),
            'areas': torch.tensor([50.0] * 6),
            'target': 25.0,
            'plot_group': 'plot_B',
            'split': 'train'
        },
        {
            'idx': 2,
            'filename': 'img3.jpg',
            'features': torch.randn(3, 384),
            'areas': torch.tensor([100.0] * 3),
            'target': 10.0,
            'plot_group': 'plot_C',
            'split': 'val'
        },
        {
            'idx': 3,
            'filename': 'img4.jpg',
            'features': torch.randn(2, 384),
            'areas': torch.tensor([80.0] * 2),
            'target': 40.0,
            'plot_group': 'plot_D',
            'split': 'test'
        }
    ]
    torch.save({'bags': bags}, cache_file)
    return str(cache_file)

def test_csfb_cached_bag_dataset(mock_cache_file):
    train_dataset = CSFBCachedBagDataset(mock_cache_file, split='train')
    assert len(train_dataset) == 2
    
    feat, area, target, group = train_dataset[0]
    assert feat.shape == (4, 384)
    assert area.shape == (4,)
    assert target.item() == 12.5
    assert group == 'plot_A'

def test_csfb_cached_paired_bag_dataset(mock_cache_file):
    paired_dataset = CSFBCachedPairedBagDataset(mock_cache_file, split='train', margin=5.0)
    assert len(paired_dataset) == 2
    
    fA, aA, tA, gA, fB, aB, tB, gB = paired_dataset[0]
    assert fA.shape[1] == 384
    assert fB.shape[1] == 384
    assert abs(tA.item() - tB.item()) >= 5.0

def test_get_patch_dataloaders_cached(mock_cache_file, tmp_path):
    manifest_file = tmp_path / "mock_manifest.csv"
    manifest_file.write_text("filename,mean_score,plot_group,split\n")
    
    train_loader, val_loader, test_loader = get_patch_dataloaders(
        manifest_path=str(manifest_file),
        batch_size=2,
        cache_path=mock_cache_file
    )
    
    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == 1
    assert len(test_loader.dataset) == 1
    
    batch_features, batch_areas, batch_targets, batch_groups = next(iter(val_loader))
    assert len(batch_features) == 1
    assert batch_features[0].shape == (3, 384)
    assert batch_targets[0].item() == 10.0
