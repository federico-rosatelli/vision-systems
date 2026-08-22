import os
import glob
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def parse_score_csv(csv_path):
    """
    Parses a single score CSV robustly.
    Handles different separators (; or ,), BOM characters, and cleans filenames.
    """
    try:
        # engine='python' and sep=None allow pandas to guess the separator
        df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8-sig')
        
        # Look for the column containing the filename
        filename_col = [c for c in df.columns if 'file' in c.lower() or 'image' in c.lower() or 'picture' in c.lower()]
        
        if filename_col:
            col = filename_col[0]
            # Ensure all filenames end with .jpg
            df[col] = df[col].astype(str).apply(
                lambda x: x if x.lower().endswith('.jpg') else f"{x}.jpg"
            )
            df.rename(columns={col: 'filename'}, inplace=True)
            
        return df
    except Exception as e:
        print(f"Error parsing {csv_path}: {e}")
        return pd.DataFrame()

def build_manifest(raw_data_dir, output_manifest_path):
    """
    Scans the raw_data_dir, parses the CSVs and builds a unified manifest.
    (Implementation of Phase 0 from PROJECT_PLAN.md)
    """
    raw_dir = Path(raw_data_dir)
    
    # 1. Find all CSV files
    csv_files = list(raw_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files.")
    
    if not csv_files:
        print("No CSV files found. Check the data path.")
        return
    
    # 2. Merge all dataframes
    all_dfs = []
    for csv_file in csv_files:
        df = parse_score_csv(csv_file)
        if not df.empty:
            df['source_csv'] = csv_file.name
            all_dfs.append(df)
            
    if not all_dfs:
        print("No data extracted from CSVs.")
        return
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # 3. Column Standardization (Based on project assumptions)
    jlu_col = next((c for c in master_df.columns if 'JLU' in c.upper() and 'SCORE' in c.upper()), None)
    gau_col = next((c for c in master_df.columns if 'GAU' in c.upper() and 'SCORE' in c.upper()), None)
    
    if jlu_col and gau_col:
        print(f"Found score columns: {jlu_col} and {gau_col}")
        master_df['Score_JLU'] = pd.to_numeric(master_df[jlu_col], errors='coerce')
        master_df['Score_GAU'] = pd.to_numeric(master_df[gau_col], errors='coerce')
        
        # Calculate rater disagreement and mean
        master_df['disagreement'] = (master_df['Score_JLU'] - master_df['Score_GAU']).abs()
        master_df['mean_score'] = (master_df['Score_JLU'] + master_df['Score_GAU']) / 2.0
        
        # High quality images: disagreement less than 5%
        master_df['is_high_quality'] = master_df['disagreement'] < 5.0
    else:
        print("Warning: Score columns for JLU and GAU not identified accurately. Current columns are:")
        print(master_df.columns.tolist())
        master_df['is_high_quality'] = False
        master_df['mean_score'] = np.nan
        master_df['disagreement'] = np.nan
        
    # 4. Extract metadata from filenames (Date, BBCH)
    def extract_metadata(filename):
        # Basic extraction based on assumed filename structure (e.g., 2025_10_14_RSFB-Phenotyping_GG_...)
        parts = str(filename).replace('.jpg', '').split('_')
        date = None
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
            date = f"{parts[0]}-{parts[1]}-{parts[2]}"
        
        # Extract BBCH if present
        bbch = next((p for p in parts if 'BBCH' in p.upper()), "Unknown")
        return pd.Series({'extracted_date': date, 'extracted_bbch': bbch})
        
    if 'filename' in master_df.columns:
        metadata = master_df['filename'].apply(extract_metadata)
        master_df = pd.concat([master_df, metadata], axis=1)
        
        # 5. Verify physical existence of images
        print("Scanning for physical .jpg files...")
        image_paths = {p.name: str(p) for p in raw_dir.rglob("*.jpg")}
        master_df['image_path'] = master_df['filename'].map(image_paths)
        master_df['file_exists'] = master_df['image_path'].notna()
        
    # 6. Plot Grouping Handling (Leakage Prevention)
    group_col = next((c for c in master_df.columns if c.lower() in ['qr-code', 'qr_code', 'plot_id', 'plotnr']), None)
    if not group_col:
        master_df['plot_group'] = 'requires_qr_extraction'
    else:
        master_df['plot_group'] = master_df[group_col].fillna('unknown').astype(str)

    # Save the manifest
    output_path = Path(output_manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_path, index=False)
    
    print(f"\nData manifest successfully generated at: {output_path}")
    
    # Final Audit Report
    print("\n--- Audit Report (Phase 0) ---")
    print(f"Total rows found in CSVs: {len(master_df)}")
    if 'file_exists' in master_df.columns:
        print(f"Images actually found on disk: {master_df['file_exists'].sum()}")
    print(f"High Quality images (diff < 5): {master_df['is_high_quality'].sum()}")
    print(f"Mean disagreement between raters: {master_df['disagreement'].mean():.2f}")
    
