import unittest
import torch
from src.models.dinov3_regressor import DINOv3Regressor, get_loss_function

class TestModels(unittest.TestCase):
    def setUp(self):
        # Initialize model with placeholder dinov2_vits14 for testing without internet dependency issues
        self.model = DINOv3Regressor(model_name='dinov2_vits14', head_width=64, dropout_p=0.1)
        
    def test_frozen_backbone(self):
        """Ensure backbone requires no gradients and head requires gradients."""
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                self.assertFalse(param.requires_grad, f"Parameter {name} in backbone should not require grad.")
            if 'regression_head' in name:
                self.assertTrue(param.requires_grad, f"Parameter {name} in head should require grad.")
                
    def test_output_bounds(self):
        """Ensure that the model output is strictly between 0 and 100."""
        self.model.eval()
        dummy_input = torch.randn(4, 3, 224, 224)
        
        with torch.no_grad():
            output = self.model(dummy_input)
            
        self.assertEqual(output.shape, (4,), "Output shape must be (batch_size,)")
        self.assertTrue(torch.all(output >= 0), "Output has values < 0")
        self.assertTrue(torch.all(output <= 100), "Output has values > 100")
        
    def test_loss_functions(self):
        """Ensure get_loss_function returns correct criterion."""
        huber = get_loss_function('huber')
        mse = get_loss_function('mse')
        
        self.assertIsInstance(huber, torch.nn.HuberLoss)
        self.assertIsInstance(mse, torch.nn.MSELoss)
        
        with self.assertRaises(ValueError):
            get_loss_function('unknown')

if __name__ == '__main__':
    unittest.main()
