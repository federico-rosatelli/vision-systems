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
        # We divide the margin by 100 if targets are [0, 100]? 
        # Actually our targets are [0, 100], so margin=5 means 5% difference.
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.lambda_rank = lambda_rank
        
    def forward(self, pred_A, pred_B, target_A, target_B):
        # 1. Regression Loss on both items
        loss_reg_A = self.regression_loss(pred_A, target_A)
        loss_reg_B = self.regression_loss(pred_B, target_B)
        loss_reg = (loss_reg_A + loss_reg_B) / 2.0
        
        # 2. Ranking Loss
        # target_rank should be 1 if A > B, and -1 if A < B. 
        # If they are equal (which shouldn't happen with our margin filter), we can set it to 0, 
        # but MarginRankingLoss expects 1 or -1.
        target_rank = torch.where(target_A > target_B, 
                                  torch.ones_like(target_A), 
                                  -torch.ones_like(target_B))
                                  
        loss_rank = self.ranking_loss(pred_A, pred_B, target_rank)
        
        # 3. Joint Loss
        return loss_reg + self.lambda_rank * loss_rank
