import pytest
import torch
from unittest.mock import MagicMock, patch
from src.models.patch_model import DINOv3PatchRegressor

@patch("src.models.patch_model.AutoModel.from_pretrained")
def test_patch_model_weighted_aggregation(mock_from_pretrained):
    # Mock the backbone to return predictable features
    mock_backbone = MagicMock()
    mock_hidden_size = 384
    mock_backbone.config.hidden_size = mock_hidden_size
    mock_from_pretrained.return_value = mock_backbone
    
    model = DINOv3PatchRegressor(aggregation="weighted")

    
    # We provide a mock feature output for patches
    # Batch size 2. 
    # Image 1: 2 patches. Area 1: 100, Area 2: 300 (weights: 0.25, 0.75)
    # Patch 1 feature: all 1.0, Patch 2 feature: all 5.0
    # Expected weighted feature: 1.0 * 0.25 + 5.0 * 0.75 = 0.25 + 3.75 = 4.0
    
    # Image 2: 1 patch. Area: 100
    # Patch 1 feature: all 2.0
    # Expected weighted feature: 2.0
    
    def mock_extract(patches):
        if patches.shape[0] == 2:
            feat = torch.ones(2, mock_hidden_size)
            feat[1, :] = 5.0
            return feat
        elif patches.shape[0] == 1:
            return torch.ones(1, mock_hidden_size) * 2.0
            
    model._extract_patch_features = mock_extract
    
    # Create inputs
    # The actual patch tensor content doesn't matter since we mock extraction
    patches_img1 = torch.zeros(2, 3, 224, 224)
    patches_img2 = torch.zeros(1, 3, 224, 224)
    patch_tensors_list = [patches_img1, patches_img2]
    
    areas_img1 = torch.tensor([100.0, 300.0])
    areas_img2 = torch.tensor([100.0])
    area_tensors_list = [areas_img1, areas_img2]
    
    class MockHead(torch.nn.Module):
        def forward(self, x):
            return x[:, 0:1]
    model.head = MockHead()
    
    output = model(patch_tensors_list, area_tensors_list)
    # Model multiplies head output by 100.0
    # Expected: [4.0 * 100.0, 2.0 * 100.0] = [400.0, 200.0]
    
    assert output.shape == (2, 1)
    assert torch.allclose(output[0], torch.tensor([400.0]))
    assert torch.allclose(output[1], torch.tensor([200.0]))

@patch("src.models.patch_model.AutoModel.from_pretrained")
def test_patch_model_uniform_aggregation(mock_from_pretrained):
    mock_backbone = MagicMock()
    mock_hidden_size = 384
    mock_backbone.config.hidden_size = mock_hidden_size
    mock_from_pretrained.return_value = mock_backbone
    
    model = DINOv3PatchRegressor(aggregation="uniform")
    
    # Image 1: 2 patches. Patch 1 feature: 1.0, Patch 2 feature: 5.0
    # Expected uniform feature: (1.0 + 5.0) / 2 = 3.0
    
    def mock_extract(patches):
        feat = torch.ones(2, mock_hidden_size)
        feat[1, :] = 5.0
        return feat
            
    model._extract_patch_features = mock_extract
    
    patch_tensors_list = [torch.zeros(2, 3, 224, 224)]
    area_tensors_list = [torch.tensor([100.0, 300.0])]
    
    class MockHead(torch.nn.Module):
        def forward(self, x):
            return x[:, 0:1]
    model.head = MockHead()
    
    output = model(patch_tensors_list, area_tensors_list)
    # Expected: 3.0 * 100.0 = 300.0
    
    assert torch.allclose(output[0], torch.tensor([300.0]))

@patch("src.models.patch_model.AutoModel.from_pretrained")
def test_patch_model_zero_patches(mock_from_pretrained):
    mock_backbone = MagicMock()
    mock_hidden_size = 384
    mock_backbone.config.hidden_size = mock_hidden_size
    mock_from_pretrained.return_value = mock_backbone
    
    model = DINOv3PatchRegressor(aggregation="weighted")
    
    # If 0 patches passed, dataset fallback actually returns 1 dummy patch with 0 area
    patch_tensors_list = [torch.zeros(1, 3, 224, 224)]
    area_tensors_list = [torch.tensor([0.0])]
    
    def mock_extract(patches):
        return torch.ones(1, mock_hidden_size) * 7.0
        
    model._extract_patch_features = mock_extract
    class MockHead(torch.nn.Module):
        def forward(self, x):
            return x[:, 0:1]
    model.head = MockHead()
    
    output = model(patch_tensors_list, area_tensors_list)
    # Sum of areas is 0, should fallback to uniform mean of the 1 patch -> 7.0
    # output -> 7.0 * 100.0 = 700.0
    
    assert torch.allclose(output[0], torch.tensor([700.0]))
