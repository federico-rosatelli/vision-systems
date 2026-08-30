import torch
import torch.nn as nn
from transformers import AutoModel

class DINOv3PatchRegressor(nn.Module):
    """
    Model that aggregates features from multiple plant patches per image.
    Uses frozen DINOv3 as feature extractor.
    """
    def __init__(self, model_name="facebook/dinov3-vits16-pretrain-lvd1689m", head_width=256, dropout_p=0.3, aggregation="weighted", local_weights_path=None):
        super().__init__()
        
        self.aggregation = aggregation # "weighted" or "uniform"
        
        # Load backbone
        if local_weights_path:
            self.backbone = AutoModel.from_pretrained(local_weights_path)
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
            
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # DINOv3 ViT-S/16 has hidden size 384
        embed_dim = self.backbone.config.hidden_size
        
        # Regression Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_width),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(head_width, 1),
            nn.Sigmoid()
        )
        
    def _extract_patch_features(self, patches):
        """ Extract features for a [N, 3, 224, 224] tensor of patches """
        if patches.shape[0] == 0:
            return torch.zeros(1, self.backbone.config.hidden_size, device=patches.device)
            
        outputs = self.backbone(pixel_values=patches)
        # Use [CLS] token representation for each patch
        return outputs.last_hidden_state[:, 0, :]
        
    def forward(self, patch_tensors_list, area_tensors_list):
        """
        Args:
            patch_tensors_list: list of [N_i, 3, 224, 224] tensors for each image in batch
            area_tensors_list: list of [N_i] tensors with visible areas
        Returns:
            [B, 1] tensor of predicted scores (0-100)
        """
        batch_size = len(patch_tensors_list)
        aggregated_features = []
        
        for i in range(batch_size):
            patches = patch_tensors_list[i]
            areas = area_tensors_list[i]
            
            features = self._extract_patch_features(patches) # [N, D]
            
            if self.aggregation == "weighted" and areas.sum() > 0:
                # Normalize areas to sum to 1
                weights = areas / areas.sum()
                # Multiply features by weights and sum over N
                # weights shape: [N], features shape: [N, D]
                agg_feat = (features * weights.unsqueeze(1)).sum(dim=0)
            else:
                # Uniform mean (fallback if sum == 0 or explicitly requested)
                agg_feat = features.mean(dim=0)
                
            aggregated_features.append(agg_feat)
            
        # Stack into [B, D]
        batch_features = torch.stack(aggregated_features)
        
        # Pass through regression head and scale to 0-100
        output = self.head(batch_features) * 100.0
        return output
