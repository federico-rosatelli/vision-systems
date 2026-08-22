import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_splits(manifest_path, output_path, random_state=42, test_size=0.15, val_size=0.15):
    """
    Reads the data manifest and creates group-aware, score-stratified 
    train/validation/test splits.
    Ensures images from the same plot_group stay in the same split.
    """
    print(f"Loading manifest from {manifest_path}...")
    df = pd.read_csv(manifest_path)
    
    if 'mean_score' not in df.columns or 'plot_group' not in df.columns:
        print("Error: Manifest must contain 'mean_score' and 'plot_group' columns.")
        return
        
    # Drop rows without a valid score to compute stratification
    valid_df = df.dropna(subset=['mean_score']).copy()
    
    if len(valid_df) == 0:
        print("Error: No valid scores found to perform stratification.")
        return
        
    print(f"Total valid images for splitting: {len(valid_df)}")
    
    # 1. Aggregate scores by plot_group
    group_stats = valid_df.groupby('plot_group')['mean_score'].mean().reset_index()
    
    # 2. Create score bins for stratification
    # We bin the continuous damage score into discrete categories
    bins = [-1, 5, 10, 20, 40, 101]
    labels = [0, 1, 2, 3, 4]
    group_stats['score_bin'] = pd.cut(group_stats['mean_score'], bins=bins, labels=labels)
    
    # Fallback for NaN bins if any score is out of bounds
    group_stats['score_bin'] = group_stats['score_bin'].fillna(0)
    
    unique_groups = group_stats['plot_group'].values
    group_bins = group_stats['score_bin'].values
    
    # First split: Train vs Temp (Val + Test)
    temp_size = test_size + val_size
    # Handle the second split properly based on proportion
    val_ratio_in_temp = val_size / temp_size if temp_size > 0 else 0.5
    
    try:
        train_groups, temp_groups, _, temp_bins = train_test_split(
            unique_groups, group_bins, test_size=temp_size, 
            random_state=random_state, stratify=group_bins
        )
        
        # Second split: Val vs Test
        val_groups, test_groups = train_test_split(
            temp_groups, test_size=(1.0 - val_ratio_in_temp), 
            random_state=random_state, stratify=temp_bins
        )
        print("Successfully performed stratified group splitting.")
    except ValueError as e:
        print(f"Warning: Stratification failed ({e}). Falling back to random group split.")
        train_groups, temp_groups = train_test_split(
            unique_groups, test_size=temp_size, random_state=random_state
        )
        val_groups, test_groups = train_test_split(
            temp_groups, test_size=(1.0 - val_ratio_in_temp), random_state=random_state
        )
        
    # 4. Map the assigned splits back to the original dataframe
    # Initialize all as 'unassigned'
    df['split'] = 'unassigned'
    
    df.loc[df['plot_group'].isin(train_groups), 'split'] = 'train'
    df.loc[df['plot_group'].isin(val_groups), 'split'] = 'val'
    df.loc[df['plot_group'].isin(test_groups), 'split'] = 'test'
    
    # Audit the splits
    print("\n--- Split Distribution (Images) ---")
    print(df['split'].value_counts(normalize=True).apply(lambda x: f"{x*100:.1f}%"))
    print("\n--- Split Distribution (Plots) ---")
    print(df.groupby('split')['plot_group'].nunique())
    
    # 5. Save fixed split IDs for reproducibility (Phase 1 requirement)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nManifest with splits saved to {output_path}")

