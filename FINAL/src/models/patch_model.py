import torch
import torch.nn as nn
from transformers import AutoModel
from src.models.mil_model import DINOv3MILRegressor, AttentionMIL, GatedAttentionMIL

class DINOv3PatchRegressor(nn.Module):
    """
    Model that aggregates features from multiple plant patches per image.
    Uses frozen DINOv3 as feature extractor.
    Supports aggregation modes: 'abmil', 'gated_abmil', 'weighted', 'uniform'.
    """
    def __init__(self, model_name="facebook/dinov3-vits16-pretrain-lvd1689m", head_width=256, dropout_p=0.3, aggregation="weighted", local_weights_path=None, attn_L=128, image_size=224, **kwargs):
        super().__init__()
        
        self.aggregation = aggregation
        self.mil_regressor = DINOv3MILRegressor(
            model_name=model_name,
            head_width=head_width,
            dropout_p=dropout_p,
            aggregation=aggregation,
            local_weights_path=local_weights_path,
            attn_L=attn_L,
            image_size=image_size,
            **kwargs
        )
        self.backbone = self.mil_regressor.backbone
        self.head = self.mil_regressor.head

    def _extract_patch_features(self, patches):
        return self.mil_regressor.extract_patch_features(patches)

    def forward(self, patch_tensors_list, area_tensors_list=None):
        """
        Args:
            patch_tensors_list: list of [N_i, 3, 224, 224] image patch tensors OR [N_i, D] feature tensors
            area_tensors_list: list of [N_i] tensors with visible areas
        Returns:
            [B, 1] tensor of predicted scores (0-100)
        """
        # Sync dynamically attached properties/submodules from tests or wrappers
        if hasattr(self, '_modules') and 'head' in self._modules:
            self.mil_regressor.head = self._modules['head']
        if hasattr(self, 'aggregation'):
            self.mil_regressor.aggregation = self.aggregation
        self.mil_regressor.extract_patch_features = self._extract_patch_features

        return self.mil_regressor(patch_tensors_list, area_tensors_list)
