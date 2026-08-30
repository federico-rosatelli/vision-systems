import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np

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
        
        if 'image_path' in row and pd.notna(row['image_path']):
            img_path = row['image_path']
        elif 'absolute_path' in row and pd.notna(row['absolute_path']):
            img_path = row['absolute_path']
        else:
            img_path = row['path'] if 'path' in row else row['filename']
            
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
        # If no plants detected, return a dummy patch to avoid crashing
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
                # Convert BGR to RGB for PIL/Torch
                patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                patch_pil = Image.fromarray(patch_rgb)
                
                if self.transform:
                    patch_tensors.append(self.transform(patch_pil))
                else:
                    patch_tensors.append(transforms.ToTensor()(patch_pil))
                    
        # Stack into [N, C, H, W]
        patch_tensor = torch.stack(patch_tensors)
        area_tensor = torch.tensor(areas, dtype=torch.float32)
        
        score = float(row['mean_score'])
        target = torch.tensor(score, dtype=torch.float32)
        plot_group = str(row.get('plot_group', 'unknown'))
        
        return patch_tensor, area_tensor, target, plot_group


def patch_collate_fn(batch):
    """
    Custom collate function for CSFBPlantPatchDataset.
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


def get_patch_dataloaders(manifest_path, batch_size=32, num_workers=4, image_size=224, high_quality_only=True):
    """
    Creates and returns train, validation, and test dataloaders for the patch-based model.
    """
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
    
    train_dataset = CSFBPlantPatchDataset(
        manifest_path, split='train', transform=train_transform, high_quality_only=high_quality_only
    )
        
    val_dataset = CSFBPlantPatchDataset(
        manifest_path, split='val', transform=eval_transform, high_quality_only=high_quality_only
    )
    
    test_dataset = CSFBPlantPatchDataset(
        manifest_path, split='test', transform=eval_transform, high_quality_only=high_quality_only
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, drop_last=True, collate_fn=patch_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=patch_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=patch_collate_fn
    )
    
    return train_loader, val_loader, test_loader
