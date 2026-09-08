import os
import argparse
import time
import torch
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.patch_dataset import CSFBPlantPatchDataset, patch_collate_fn
from src.models.mil_model import DINOv3MILRegressor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def cache_dinov3_patch_embeddings(
    manifest_path="outputs/tables/data_manifest_split.csv",
    output_cache_path="outputs/cache/dinov3_bags.pt",
    weights_path="weights/dinov3-vits16-hf",
    model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    batch_size=1,
    num_workers=4,
    image_size=224
):
    """
    Extracts DINOv3 patch embeddings for every image in manifest_path and saves them
    into a PyTorch pickle dictionary file for 100x faster MIL training iterations.
    """
    os.makedirs(os.path.dirname(output_cache_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Caching patch embeddings using device: {device}")

    # Load Model (only backbone needed)
    local_weights = weights_path if os.path.exists(weights_path) else None
    model = DINOv3MILRegressor(
        model_name=model_name,
        local_weights_path=local_weights
    )
    model.to(device)
    model.eval()

    # Define transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

    # Load Dataset
    dataset = CSFBPlantPatchDataset(
        manifest_path=manifest_path,
        split=None, # Load all splits
        transform=eval_transform,
        high_quality_only=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=patch_collate_fn)

    print(f"Extracting features for {len(dataset)} images...")
    cached_bags = []

    start_t = time.time()
    with torch.no_grad():
        for idx, (patch_tensors, area_tensors, targets, plot_groups) in enumerate(tqdm(loader, desc="Extracting DINOv3 Features")):
            row = dataset.df.iloc[idx]
            filename = row['filename']
            split = row.get('split', 'train')

            patches = patch_tensors[0].to(device) # [N, 3, 224, 224]
            areas = area_tensors[0].cpu() # [N]
            target = targets[0].item()
            plot_group = plot_groups[0]

            # Extract DINOv3 patch features
            features = model.extract_patch_features(patches).cpu() # [N, 384]

            cached_bags.append({
                'idx': idx,
                'filename': filename,
                'features': features, # [N, 384] CPU tensor
                'areas': areas,       # [N] CPU tensor
                'target': target,     # float
                'plot_group': plot_group,
                'split': split
            })

    elapsed = time.time() - start_t
    print(f"Extraction complete in {elapsed:.1f}s.")

    torch.save({
        'version': 1,
        'model_name': model_name,
        'weights_path': weights_path,
        'manifest_path': manifest_path,
        'bags': cached_bags
    }, output_cache_path)

    print(f"Cached {len(cached_bags)} feature bags to {output_cache_path}")
    return output_cache_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache DINOv3 Patch Embeddings")
    parser.add_argument("--manifest", default="outputs/tables/data_manifest_split.csv")
    parser.add_argument("--out_cache", default="outputs/cache/dinov3_bags.pt")
    parser.add_argument("--weights_path", default="weights/dinov3-vits16-hf")
    args = parser.parse_args()
    cache_dinov3_patch_embeddings(args.manifest, args.out_cache, weights_path=args.weights_path)
