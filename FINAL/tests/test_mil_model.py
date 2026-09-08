import pytest
import torch
from unittest.mock import MagicMock, patch
from src.models.mil_model import AttentionMIL, GatedAttentionMIL, DINOv3MILRegressor

def test_attention_mil_forward():
    mil_pool = AttentionMIL(in_features=384, L=128)
    
    # 1. Single Bag input [N, 384]
    features = torch.randn(7, 384)
    bag_feat, attn = mil_pool(features)
    
    assert bag_feat.shape == (384,)
    assert attn.shape == (7, 1)
    assert torch.allclose(attn.sum(), torch.tensor(1.0), atol=1e-5)
    assert (attn >= 0.0).all() and (attn <= 1.0).all()

def test_gated_attention_mil_forward():
    gated_pool = GatedAttentionMIL(in_features=384, L=128)
    
    # Single Bag input [N, 384]
    features = torch.randn(5, 384)
    bag_feat, attn = gated_pool(features)
    
    assert bag_feat.shape == (384,)
    assert attn.shape == (5, 1)
    assert torch.allclose(attn.sum(), torch.tensor(1.0), atol=1e-5)
    assert (attn >= 0.0).all() and (attn <= 1.0).all()

def test_dinov3_mil_regressor_cached_inputs():
    model = DINOv3MILRegressor(
        model_name=None,
        local_weights_path=None,
        head_width=64,
        aggregation="abmil",
        embed_dim=384,
        attn_L=32
    )
    model.eval()
    
    # Pass a batch of pre-extracted feature bags
    bag1 = torch.randn(4, 384)
    bag2 = torch.randn(10, 384)
    patch_inputs = [bag1, bag2]
    
    scores = model(patch_inputs)
    assert scores.shape == (2, 1)
    assert (scores >= 0.0).all() and (scores <= 100.0).all()

def test_dinov3_mil_regressor_gated_cached_inputs():
    model = DINOv3MILRegressor(
        model_name=None,
        local_weights_path=None,
        head_width=64,
        aggregation="gated_abmil",
        embed_dim=384,
        attn_L=32
    )
    model.eval()
    
    bag1 = torch.randn(3, 384)
    bag2 = torch.randn(6, 384)
    patch_inputs = [bag1, bag2]
    
    scores = model(patch_inputs)
    assert scores.shape == (2, 1)
    assert (scores >= 0.0).all() and (scores <= 100.0).all()

@patch("src.models.mil_model.AutoModel.from_pretrained")
def test_dinov3_mil_regressor_image_inputs(mock_from_pretrained):
    mock_backbone = MagicMock()
    mock_backbone.config.hidden_size = 384
    mock_from_pretrained.return_value = mock_backbone
    
    model = DINOv3MILRegressor(
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        head_width=64,
        aggregation="abmil"
    )
    
    def mock_extract(patches):
        return torch.ones(patches.shape[0], 384)
        
    model.extract_patch_features = mock_extract
    
    # Raw image patches [N, 3, 224, 224]
    patches_bag1 = torch.randn(2, 3, 224, 224)
    patches_bag2 = torch.randn(3, 3, 224, 224)
    
    scores = model([patches_bag1, patches_bag2])
    assert scores.shape == (2, 1)
    assert (scores >= 0.0).all() and (scores <= 100.0).all()
