import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel

class DINOv3Regressor(nn.Module):
    """
    DINOv3 Baseline model with frozen backbone and trainable MLP regression head.
    Implements Phase 2 of the project plan.
    """
    def __init__(self, model_name, head_width=256, dropout_p=0.3, image_size=224,
                 output_range=(0.0, 100.0), weights_path=None):
        """
        Args:
            model_name (str): The name of the DINO checkpoint to load.
            head_width (int): Number of hidden units in the MLP head.
            dropout_p (float): Dropout probability for the MLP head to prevent overfitting.
        """
        super().__init__()
        
        if not model_name.startswith("dinov3_"):
            raise ValueError("Official runs require an explicit DINOv3 model name")
        self.model_name = model_name
        self.image_size = image_size
        if list(output_range) != [0.0, 100.0]:
            raise ValueError("Only the fixed damage output range [0, 100] is supported")
        if not weights_path:
            raise ValueError(
                "weights_path must point to the downloaded Hugging Face DINOv3 directory"
            )
        weights_path = Path(weights_path).expanduser().resolve()
        if not (weights_path / "config.json").is_file():
            raise FileNotFoundError(f"DINOv3 config not found in: {weights_path}")
        if not (weights_path / "model.safetensors").is_file():
            raise FileNotFoundError(f"DINOv3 weights not found in: {weights_path}")

        self.weights_path = str(weights_path)
        print(f"Loading explicit DINOv3 backbone from {weights_path}...")
        try:
            self.backbone = AutoModel.from_pretrained(
                weights_path, local_files_only=True
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load required DINOv3 backbone from {weights_path}; "
                "official runs do not fall back to another model"
            ) from error
            
        # Freeze the backbone completely (Phase 2 requirement)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.backbone.eval() # Ensure dropout/batchnorm in backbone are disabled
            
        embed_dim = self.backbone.config.hidden_size
            
        print(f"Backbone frozen. Extracted embedding dimension: {embed_dim}")
        
        # MLP Regression Head
        # Tunable parameters: head_width, dropout_p
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, head_width),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(head_width, 1)
        )
        
    def forward(self, x):
        """
        Forward pass.
        Returns a bounded score between 0 and 100 representing the damage percentage.
        """
        # Ensure backbone stays in eval mode and no gradients are tracked for it
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(pixel_values=x).pooler_output
            
        # Pass features through the trainable regression head
        logits = self.regression_head(features)
        
        # Bound the prediction to [0, 100] using Sigmoid as specified in the plan
        bounded_score = 100.0 * torch.sigmoid(logits)
        
        # Squeeze the last dimension to match target shapes: (batch_size,)
        return bounded_score.squeeze(-1)

def get_loss_function(loss_type='huber', delta=1.0):
    """
    Utility to get the regression loss function.
    """
    if loss_type.lower() == 'huber':
        return nn.HuberLoss(delta=delta)
    elif loss_type.lower() == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError("Unsupported loss_type. Choose 'huber' or 'mse'.")
