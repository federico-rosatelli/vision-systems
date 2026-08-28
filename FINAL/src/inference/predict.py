import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.dinov3_regressor import DINOv3Regressor

def predict_single_image(image_path, model_path, image_size=None):
    """
    Predicts the CSFB damage score for a single image.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('model_config')
    if not model_config:
        raise ValueError("Checkpoint lacks required model_config metadata")
    checkpoint_image_size = model_config['image_size']
    if image_size is not None and image_size != checkpoint_image_size:
        raise ValueError("Requested image size does not match the checkpoint metadata")
    image_size = checkpoint_image_size

    # Same standard transform used in evaluation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                                     
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    
    model = DINOv3Regressor(**model_config)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        score = model(tensor).item()
        
    print(f"=====================================")
    print(f" Image: {os.path.basename(image_path)}")
    print(f" Predicted CSFB Damage: {score:.2f} %")
    print(f"=====================================")
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image to analyze")
    parser.add_argument("--model", type=str, default="outputs/runs/baseline_regression_seed42/checkpoints/best_model.pth", help="Path to best_model.pth")
    args = parser.parse_args()
    
    predict_single_image(args.image, args.model)
