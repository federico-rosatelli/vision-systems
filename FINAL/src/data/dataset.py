import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CSFBDataset(Dataset):
    """
    PyTorch Dataset for Cabbage Stem Flea Beetle (CSFB) feeding damage quantification.
    Reads data from the consolidated manifest CSV.
    """
    def __init__(self, manifest_path, split=None, transform=None, high_quality_only=False):
        """
        Args:
            manifest_path (str): Path to the generated data_manifest.csv.
            split (str): 'train', 'val', or 'test'. Filters the dataframe if 'split' column exists.
            transform (callable, optional): Optional transform to be applied on an image.
            high_quality_only (bool): If True, strictly filters for is_high_quality == True (diff < 5).
        """
        self.df = pd.read_csv(manifest_path)
        
        # Filter for existing files
        if 'file_exists' in self.df.columns:
            self.df = self.df[self.df['file_exists'] == True]
            
        # Optional: filter by split
        if split and 'split' in self.df.columns:
            self.df = self.df[self.df['split'] == split]
            
        # Optional: filter for baseline requirement (470 high quality images)
        if high_quality_only and 'is_high_quality' in self.df.columns:
            self.df = self.df[self.df['is_high_quality'] == True]
            
        # Drop rows with missing target scores
        self.df = self.df.dropna(subset=['mean_score'])
        self.df = self.df.reset_index(drop=True)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Determine image path
        if 'image_path' in row and pd.notna(row['image_path']):
            img_path = row['image_path']
        elif 'absolute_path' in row and pd.notna(row['absolute_path']):
            # Fallback for old manifests
            img_path = row['absolute_path']
        else:
            img_path = row['path'] if 'path' in row else row['filename']
            
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Extract target score. The score is between 0 and 100.
        score = float(row['mean_score'])
        target = torch.tensor(score, dtype=torch.float32)
        
        # Return plot_group as well to handle aggregation later
        plot_group = str(row.get('plot_group', 'unknown'))
        
        return image, target, plot_group


class CSFBPairedDataset(Dataset):
    """
    Dataset that returns pairs of images for Ranking-Based Weak Supervision.
    Only creates pairs where the absolute difference in damage score > margin.
    """
    def __init__(self, manifest_path, split='train', transform=None, high_quality_only=False, margin=5.0):
        # We reuse CSFBDataset to handle the filtering
        self.base_dataset = CSFBDataset(manifest_path, split=split, transform=transform, high_quality_only=high_quality_only)
        self.transform = transform
        self.margin = margin
        self.partners = self._build_valid_partners()
        
    def _build_valid_partners(self):
        import random
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
            
        img_A, target_A, group_A = self.base_dataset[idx_A]
        img_B, target_B, group_B = self.base_dataset[idx_B]
        return img_A, img_B, target_A, target_B, group_A, group_B

def get_dataloaders(manifest_path, batch_size=32, num_workers=4, image_size=518, high_quality_only=True, training_mode='regression', joint_margin=5.0):
    """
    Creates and returns train, validation, and test dataloaders.
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
    
    if training_mode == 'joint':
        train_dataset = CSFBPairedDataset(
            manifest_path, split='train', transform=train_transform, high_quality_only=high_quality_only, margin=joint_margin
        )
    else:
        train_dataset = CSFBDataset(
            manifest_path, split='train', transform=train_transform, high_quality_only=high_quality_only
        )
        
    val_dataset = CSFBDataset(
        manifest_path, split='val', transform=eval_transform, high_quality_only=high_quality_only
    )
    test_dataset = CSFBDataset(
        manifest_path, split='test', transform=eval_transform, high_quality_only=high_quality_only
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
