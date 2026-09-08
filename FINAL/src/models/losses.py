import torch
import torch.nn as nn

class JointRankingRegressionLoss(nn.Module):
    """
    Combines Huber regression loss for absolute damage scores with 
    MarginRankingLoss for relative (pairwise) ordinal relationships.
    """
    def __init__(self, margin=5.0, lambda_rank=0.5, delta=1.0):
        super().__init__()
        self.regression_loss = nn.HuberLoss(delta=delta)
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.lambda_rank = lambda_rank
        
    def forward(self, pred_A, pred_B, target_A, target_B):
        # 1. Regression Loss on both items
        loss_reg_A = self.regression_loss(pred_A, target_A)
        loss_reg_B = self.regression_loss(pred_B, target_B)
        loss_reg = (loss_reg_A + loss_reg_B) / 2.0
        
        # 2. Ranking Loss
        target_rank = torch.where(target_A > target_B, 
                                  torch.ones_like(target_A), 
                                  -torch.ones_like(target_B))
                                  
        loss_rank = self.ranking_loss(pred_A, pred_B, target_rank)
        
        # 3. Joint Loss
        return loss_reg + self.lambda_rank * loss_rank

class AttentionEntropyLoss(nn.Module):
    """
    Computes negative entropy of attention weights to encourage uniform dispersion
    or penalize over-concentrated attention if desired.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, attn_weights_list):
        """
        attn_weights_list: list of [N_i, 1] attention tensors
        """
        entropy_sum = 0.0
        count = 0
        for attn in attn_weights_list:
            if attn is not None and attn.size(0) > 1:
                p = attn.squeeze(-1)
                entropy = -torch.sum(p * torch.log(p + self.eps))
                entropy_sum += entropy
                count += 1
        if count == 0:
            return torch.tensor(0.0, device=attn_weights_list[0].device if attn_weights_list else 'cpu')
        return entropy_sum / count
