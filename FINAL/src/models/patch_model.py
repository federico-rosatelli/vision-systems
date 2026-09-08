import torch
import torch.nn as nn
from src.models.mil_model import DINOv3MILRegressor, AttentionMIL, GatedAttentionMIL

class DINOv3PatchRegressor(nn.Module):
    """
    Model that aggregates features from multiple plant patches per image.
    Uses frozen DINOv3 as feature extractor.
    Supports aggregation modes: 'abmil', 'gated_abmil', 'weighted', 'uniform'.
    """
    def __init__(self, model_name="facebook/dinov3-vits16-pretrain-lvd1689m", head_width=256, dropout_p=0.3, aggregation="weighted", local_weights_path=None, attn_L=128):
        super().__init__()
        
        self.mil_regressor = DINOv3MILRegressor(
            model_name=model_name,
            head_width=head_width,
            dropout_p=dropout_p,
            aggregation=aggregation,
            local_weights_path=local_weights_path,
            attn_L=attn_L
        )
        self.aggregation = aggregation
        self.backbone = self.mil_regressor.backbone
        self.head = self.mil_regressor.head
        self.mil_pool = self.mil_regressor.mil_pool
        
    def forward(self, patch_tensors_list, area_tensors_list=None):
        """
        Args:
            patch_tensors_list: list of [N_i, 3, 224, 224] image patch tensors OR [N_i, D] feature tensors
            area_tensors_list: list of [N_i] tensors with visible areas
        Returns:
            [B, 1] tensor of predicted scores (0-100)
        """
        return self.mil_regressor(patch_tensors_list, area_tensors_list)
