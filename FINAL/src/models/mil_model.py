import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class AttentionMIL(nn.Module):
    """
    Standard Attention-based MIL Pooling (Ilse et al., 2018).
    Computes attention weights: a_k = softmax(w^T tanh(V h_k))
    """
    def __init__(self, in_features=384, L=128):
        super().__init__()
        self.V = nn.Linear(in_features, L)
        self.w = nn.Linear(L, 1)
        
    def forward(self, h):
        """
        Args:
            h: Tensor of shape [N, D] representing patch features for one bag.
        Returns:
            bag_feature: [D]
            A: Attention weights [N, 1]
        """
        if h.dim() == 2:
            v = torch.tanh(self.V(h)) # [N, L]
            scores = self.w(v) # [N, 1]
            A = F.softmax(scores, dim=0) # [N, 1]
            bag_feature = (h * A).sum(dim=0) # [D]
            return bag_feature, A
        else:
            v = torch.tanh(self.V(h)) # [B, N, L]
            scores = self.w(v) # [B, N, 1]
            A = F.softmax(scores, dim=1) # [B, N, 1]
            bag_feature = (h * A).sum(dim=1) # [B, D]
            return bag_feature, A

class GatedAttentionMIL(nn.Module):
    """
    Gated Attention-based MIL Pooling (Ilse et al., 2018).
    Computes attention weights: a_k = softmax(w^T (tanh(V h_k) * sigmoid(U h_k)))
    """
    def __init__(self, in_features=384, L=128):
        super().__init__()
        self.V = nn.Linear(in_features, L)
        self.U = nn.Linear(in_features, L)
        self.w = nn.Linear(L, 1)
        
    def forward(self, h):
        if h.dim() == 2:
            v = torch.tanh(self.V(h))
            u = torch.sigmoid(self.U(h))
            scores = self.w(v * u) # [N, 1]
            A = F.softmax(scores, dim=0) # [N, 1]
            bag_feature = (h * A).sum(dim=0) # [D]
            return bag_feature, A
        else:
            v = torch.tanh(self.V(h))
            u = torch.sigmoid(self.U(h))
            scores = self.w(v * u)
            A = F.softmax(scores, dim=1)
            bag_feature = (h * A).sum(dim=1)
            return bag_feature, A

class DINOv3MILRegressor(nn.Module):
    """
    MIL Regressor using DINOv3 feature embeddings and trainable aggregation module.
    Supports 'abmil', 'gated_abmil', 'weighted' (area-weighted), and 'uniform' mean.
    Can accept either raw image patch tensors [N, 3, 224, 224] or pre-extracted feature tensors [N, D].
    """
    def __init__(
        self,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        head_width=256,
        dropout_p=0.3,
        aggregation="abmil",
        local_weights_path=None,
        embed_dim=384,
        attn_L=128,
        image_size=224,
        **kwargs
    ):
        super().__init__()
        self.aggregation = aggregation
        self.embed_dim = embed_dim
        self.local_weights_path = local_weights_path
        self.model_name = model_name
        self.image_size = image_size

        # Load DINOv3 backbone if needed for image patch extraction
        self.backbone = None
        path_or_name = local_weights_path if local_weights_path else model_name
        if path_or_name:
            try:
                self.backbone = AutoModel.from_pretrained(path_or_name)
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.embed_dim = self.backbone.config.hidden_size
            except Exception as e:
                print(f"Notice: Could not load DINOv3 backbone from {path_or_name}: {e}. (Will expect pre-extracted features).")

        # Aggregation Module
        if aggregation == "abmil":
            self.mil_pool = AttentionMIL(in_features=self.embed_dim, L=attn_L)
        elif aggregation == "gated_abmil":
            self.mil_pool = GatedAttentionMIL(in_features=self.embed_dim, L=attn_L)
        else:
            self.mil_pool = None

        # Regression Head
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, head_width),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(head_width, 1),
            nn.Sigmoid()
        )

    def extract_patch_features(self, patches):
        """ Extract DINOv3 features [N, D] from patch tensor [N, 3, 224, 224] """
        if patches.shape[0] == 0:
            return torch.zeros(1, self.embed_dim, device=patches.device)
        if self.backbone is None:
            raise RuntimeError("Backbone is not loaded, cannot extract features from image patches.")
        outputs = self.backbone(pixel_values=patches)
        return outputs.last_hidden_state[:, 0, :]

    def forward_bag(self, features, areas=None):
        """
        Forward pass for a single bag of patch features [N, D].
        Returns:
            pred_score: [1, 1] tensor
            attn_weights: [N, 1] tensor
        """
        if features.shape[0] == 0:
            features = torch.zeros(1, self.embed_dim, device=features.device)

        if self.aggregation in ["abmil", "gated_abmil"]:
            agg_feat, attn_weights = self.mil_pool(features)
        elif self.aggregation == "weighted" and areas is not None and areas.sum() > 0:
            weights = areas / areas.sum()
            agg_feat = (features * weights.unsqueeze(1)).sum(dim=0)
            attn_weights = weights.unsqueeze(1)
        else:
            agg_feat = features.mean(dim=0)
            N = features.shape[0] if isinstance(features, torch.Tensor) else 1
            if not isinstance(N, int):
                N = 1
            dev = features.device if (hasattr(features, 'device') and isinstance(features.device, torch.device)) else 'cpu'
            attn_weights = torch.ones(N, 1, device=dev) / max(1, N)

        pred_score = self.head(agg_feat.unsqueeze(0)) * 100.0
        return pred_score, attn_weights

    def forward(self, patch_inputs, area_inputs=None):
        """
        Args:
            patch_inputs: List of [N_i, 3, 224, 224] image patch tensors 
                          OR list of [N_i, D] feature tensors (if cached).
            area_inputs: List of [N_i] visible area tensors (optional).
        Returns:
            scores: [B, 1] predicted damage scores (0-100)
        """
        batch_size = len(patch_inputs)
        scores = []

        for i in range(batch_size):
            inp = patch_inputs[i]
            areas = area_inputs[i] if area_inputs is not None else None

            if inp.dim() == 4:
                features = self.extract_patch_features(inp)
            else:
                features = inp

            score, _ = self.forward_bag(features, areas)
            scores.append(score)

        scores_tensor = torch.cat(scores, dim=0) # [B, 1]
        return scores_tensor
