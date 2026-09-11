import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path

from src.training.provenance import sha256_file

from src.preprocessing.frame_crop import detect_frame, crop_frame_interior, read_image_oriented
from src.preprocessing.plant_regions import extract_plant_regions

class CSFBPlantPatchDataset(Dataset):
    """
    PyTorch Dataset that returns multiple plant patches per image for CSFB damage quantification.
    Extracts patches on-the-fly using HSV color thresholding inside the metal frame.
    """
    def __init__(self, manifest_path, split=None, transform=None, hsv_bounds=None, min_plant_area=150, padding=15, high_quality_only=False):
        self.df = pd.read_csv(manifest_path)
        
        if 'file_exists' in self.df.columns:
            self.df = self.df[self.df['file_exists'] == True]
            
        if split and 'split' in self.df.columns:
            self.df = self.df[self.df['split'] == split]
            
        if high_quality_only and 'is_high_quality' in self.df.columns:
            self.df = self.df[self.df['is_high_quality'] == True]
            
        self.df = self.df.dropna(subset=['mean_score'])
        self.df = self.df.reset_index(drop=True)
        
        self.transform = transform
        self.hsv_bounds = hsv_bounds or {
            "lower": [35, 40, 40],
            "upper": [85, 255, 255]
        }
        self.min_plant_area = min_plant_area
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = None
        for col in ['image_path', 'absolute_path', 'path', 'filename']:
            if col in row and pd.notna(row[col]):
                candidate = str(row[col])
                if os.path.exists(candidate):
                    img_path = candidate
                    break
        
        if img_path is None or not os.path.exists(img_path):
            filename = str(row.get('filename', ''))
            search_dirs = [
                "../dataset/Pictures_CFSB_leaf_damage",
                "/home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage"
            ]
            for s_dir in search_dirs:
                if os.path.exists(s_dir):
                    found = list(Path(s_dir).rglob(filename))
                    if found:
                        img_path = str(found[0])
                        break

        if img_path is None or not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {row.get('filename')}")
            
        # 1. Load Image
        image_bgr = read_image_oriented(img_path)
        
        # 2. Extract Frame
        frame_det = detect_frame(image_bgr)
        if frame_det.status == "detected":
            frame_crop, _ = crop_frame_interior(image_bgr, frame_det.corners)
        else:
            frame_crop = image_bgr
            
        # 3. Detect Plants & Extract Patches
        mask, regions = extract_plant_regions(
            frame_crop,
            hue_min=self.hsv_bounds["lower"][0],
            hue_max=self.hsv_bounds["upper"][0],
            saturation_min=self.hsv_bounds["lower"][1],
            value_min=self.hsv_bounds["lower"][2],
            minimum_region_green_area=self.min_plant_area
        )
        
        patches_bgr = []
        areas = []
        for region in regions:
            x1, y1, x2, y2 = region.patch_box
            patch_bgr = frame_crop[y1:y2, x1:x2]
            patches_bgr.append(patch_bgr)
            areas.append(float(region.green_area))
        
        patch_tensors = []
        if not patches_bgr:
            dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                dummy_tensor = self.transform(dummy_image)
            else:
                dummy_tensor = transforms.ToTensor()(dummy_image)
            patch_tensors.append(dummy_tensor)
            areas = [0.0]
        else:
            for patch_bgr in patches_bgr:
                patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                patch_pil = Image.fromarray(patch_rgb)
                
                if self.transform:
                    patch_tensors.append(self.transform(patch_pil))
                else:
                    patch_tensors.append(transforms.ToTensor()(patch_pil))
                    
        patch_tensor = torch.stack(patch_tensors)
        area_tensor = torch.tensor(areas, dtype=torch.float32)
        
        score = float(row['mean_score'])
        target = torch.tensor(score, dtype=torch.float32)
        plot_group = str(row.get('plot_group', 'unknown'))
        
        return patch_tensor, area_tensor, target, plot_group


class CSFBPatchPairedDataset(Dataset):
    """
    Dataset that returns pairs of patch sets for Ranking-Based Weak Supervision.
    Only creates pairs where the absolute difference in damage score > margin.
    """
    def __init__(self, manifest_path, split='train', transform=None, hsv_bounds=None, min_plant_area=150, high_quality_only=False, margin=5.0):
        self.base_dataset = CSFBPlantPatchDataset(
            manifest_path, split=split, transform=transform, 
            hsv_bounds=hsv_bounds, min_plant_area=min_plant_area, 
            high_quality_only=high_quality_only
        )
        self.margin = margin
        self.partners = self._build_valid_partners()
        
    def _build_valid_partners(self):
        partners = {}
        df = self.base_dataset.df
        scores = df['mean_score'].values
        for i in range(len(df)):
            valid = []
            for j in range(len(df)):
                if i != j and abs(scores[i] - scores[j]) >= self.margin:
                    valid.append(j)
            partners[i] = valid
        return partners
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        import random
        idx_A = idx
        valid_partners = self.partners[idx_A]
        if not valid_partners:
            idx_B = (idx_A + 1) % len(self.base_dataset)
        else:
            idx_B = random.choice(valid_partners)
            
        patch_A, area_A, target_A, group_A = self.base_dataset[idx_A]
        patch_B, area_B, target_B, group_B = self.base_dataset[idx_B]
        return patch_A, area_A, target_A, group_A, patch_B, area_B, target_B, group_B


def validate_fixed_manifest(manifest_path, expected_count=470):
    """Validate the leakage-safe manifest used for final patch/MIL experiments."""
    df = pd.read_csv(manifest_path)
    required = {"filename", "image_path", "plot_group", "mean_score", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if len(df) != expected_count:
        raise ValueError(f"Expected {expected_count} manifest rows, found {len(df)}")
    if df["filename"].nunique(dropna=False) != expected_count:
        raise ValueError("Manifest filenames are not unique")
    if df["image_path"].nunique(dropna=False) != expected_count:
        raise ValueError("Manifest physical image paths are not unique")
    if df["plot_group"].isna().any() or df["plot_group"].astype(str).str.strip().str.lower().isin({"", "unknown", "nan"}).any():
        raise ValueError("Manifest contains missing or unknown plot groups")
    expected_splits = {"train": 331, "val": 66, "test": 73}
    actual_splits = df["split"].value_counts().to_dict()
    if actual_splits != expected_splits:
        raise ValueError(f"Expected split counts {expected_splits}, found {actual_splits}")
    group_split_counts = df.groupby("plot_group")["split"].nunique()
    if (group_split_counts > 1).any():
        raise ValueError("Manifest has plot-group leakage across splits")
    return df.reset_index(drop=True)


def validate_cache_for_manifest(cache_data, manifest_path):
    """Reject stale or incorrectly sourced embedding caches."""
    manifest_path = Path(manifest_path).resolve()
    expected_hash = sha256_file(manifest_path)
    if cache_data.get("manifest_sha256") != expected_hash:
        raise ValueError("Embedding cache does not match the requested manifest")
    manifest = pd.read_csv(manifest_path)
    bags = cache_data.get("bags", [])
    if len(bags) != len(manifest):
        raise ValueError("Embedding cache bag count does not match the manifest")
    if {b.get("filename") for b in bags} != set(manifest["filename"]):
        raise ValueError("Embedding cache filenames do not match the manifest")


class CSFBCachedBagDataset(Dataset):
    """
    PyTorch Dataset that loads pre-computed patch embedding bags from disk cache.
    """
    def __init__(self, cache_path, split=None, high_quality_only=False, manifest_path=None):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cached bag file not found: {cache_path}")
        try:
            data = torch.load(cache_path, map_location='cpu', weights_only=False)
        except TypeError:
            data = torch.load(cache_path, map_location='cpu')
        if manifest_path is not None:
            validate_cache_for_manifest(data, manifest_path)
        all_bags = data['bags']
        if high_quality_only:
            all_bags = [b for b in all_bags if b.get('is_high_quality', True)]
        if split:
            self.bags = [b for b in all_bags if b.get('split') == split]
        else:
            self.bags = all_bags
        
        self.df = pd.DataFrame([{
            'filename': b['filename'],
            'mean_score': b['target'],
            'plot_group': b['plot_group'],
            'split': b.get('split', 'unknown')
        } for b in self.bags])

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        features = bag['features'] # [N, D] tensor
        areas = bag['areas']       # [N] tensor
        target = torch.tensor(bag['target'], dtype=torch.float32)
        plot_group = str(bag['plot_group'])
        return features, areas, target, plot_group


class CSFBCachedPairedBagDataset(Dataset):
    """
    PyTorch Dataset for Paired Ranking using pre-computed patch embedding bags.
    """
    def __init__(self, cache_path, split='train', margin=5.0, high_quality_only=False, manifest_path=None):
        self.base_dataset = CSFBCachedBagDataset(
            cache_path, split=split, high_quality_only=high_quality_only,
            manifest_path=manifest_path
        )
        self.margin = margin
        self.partners = self._build_valid_partners()
        
    def _build_valid_partners(self):
        partners = {}
        df = self.base_dataset.df
        scores = df['mean_score'].values
        for i in range(len(df)):
            valid = []
            for j in range(len(df)):
                if i != j and abs(scores[i] - scores[j]) >= self.margin:
                    valid.append(j)
            partners[i] = valid
        return partners

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        import random
        idx_A = idx
        valid_partners = self.partners[idx_A]
        if not valid_partners:
            idx_B = (idx_A + 1) % len(self.base_dataset)
        else:
            idx_B = random.choice(valid_partners)
            
        feat_A, area_A, target_A, group_A = self.base_dataset[idx_A]
        feat_B, area_B, target_B, group_B = self.base_dataset[idx_B]
        return feat_A, area_A, target_A, group_A, feat_B, area_B, target_B, group_B


def patch_collate_fn(batch):
    """
    Custom collate function for CSFBPlantPatchDataset / CSFBCachedBagDataset.
    Since each image has a variable number of patches `N`, we return them as lists.
    """
    patch_tensors = []
    area_tensors = []
    targets = []
    plot_groups = []
    
    for patch_tensor, area_tensor, target, plot_group in batch:
        patch_tensors.append(patch_tensor)
        area_tensors.append(area_tensor)
        targets.append(target)
        plot_groups.append(plot_group)
        
    targets = torch.stack(targets)
    
    return patch_tensors, area_tensors, targets, plot_groups


def patch_paired_collate_fn(batch):
    """
    Custom collate function for CSFBPatchPairedDataset / CSFBCachedPairedBagDataset.
    """
    patch_tensors_A, area_tensors_A, targets_A, plot_groups_A = [], [], [], []
    patch_tensors_B, area_tensors_B, targets_B, plot_groups_B = [], [], [], []
    
    for (pA, aA, tA, gA, pB, aB, tB, gB) in batch:
        patch_tensors_A.append(pA)
        area_tensors_A.append(aA)
        targets_A.append(tA)
        plot_groups_A.append(gA)
        
        patch_tensors_B.append(pB)
        area_tensors_B.append(aB)
        targets_B.append(tB)
        plot_groups_B.append(gB)
        
    targets_A = torch.stack(targets_A)
    targets_B = torch.stack(targets_B)
    
    return patch_tensors_A, area_tensors_A, targets_A, plot_groups_A, patch_tensors_B, area_tensors_B, targets_B, plot_groups_B


def get_patch_dataloaders(manifest_path, batch_size=32, num_workers=4, image_size=224, high_quality_only=True, training_mode='regression', joint_margin=5.0, cache_path=None):
    """
    Creates and returns train, validation, and test dataloaders for the patch-based / MIL model.
    If cache_path is provided and exists, uses pre-extracted patch feature bags for 100x speedup.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading pre-extracted MIL patch feature bags from {cache_path}...")
        if training_mode == 'joint':
            train_dataset = CSFBCachedPairedBagDataset(
                cache_path, split='train', margin=joint_margin,
                high_quality_only=high_quality_only, manifest_path=manifest_path
            )
            train_collate_fn = patch_paired_collate_fn
        else:
            train_dataset = CSFBCachedBagDataset(
                cache_path, split='train', high_quality_only=high_quality_only,
                manifest_path=manifest_path
            )
            train_collate_fn = patch_collate_fn

        val_dataset = CSFBCachedBagDataset(
            cache_path, split='val', high_quality_only=high_quality_only,
            manifest_path=manifest_path
        )
        test_dataset = CSFBCachedBagDataset(
            cache_path, split='test', high_quality_only=high_quality_only,
            manifest_path=manifest_path
        )
    else:
        if cache_path:
            print(f"Notice: Cache path {cache_path} not found. Falling back to on-the-fly extraction.")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                         
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            normalize
        ])
        
        eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])
        
        if training_mode == 'joint':
            train_dataset = CSFBPatchPairedDataset(
                manifest_path, split='train', transform=train_transform, high_quality_only=high_quality_only, margin=joint_margin
            )
            train_collate_fn = patch_paired_collate_fn
        else:
            train_dataset = CSFBPlantPatchDataset(
                manifest_path, split='train', transform=train_transform, high_quality_only=high_quality_only
            )
            train_collate_fn = patch_collate_fn
            
        val_dataset = CSFBPlantPatchDataset(
            manifest_path, split='val', transform=eval_transform, high_quality_only=high_quality_only
        )
        
        test_dataset = CSFBPlantPatchDataset(
            manifest_path, split='test', transform=eval_transform, high_quality_only=high_quality_only
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers if cache_path is None else 0, drop_last=True, collate_fn=train_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers if cache_path is None else 0, collate_fn=patch_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers if cache_path is None else 0, collate_fn=patch_collate_fn
    )
    
    return train_loader, val_loader, test_loader
