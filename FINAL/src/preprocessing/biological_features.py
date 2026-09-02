import cv2
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.preprocessing.frame_crop import detect_frame, crop_frame_interior
from src.preprocessing.plant_regions import extract_plant_regions

def get_green_mask(patch_bgr, hsv_bounds=None):
    """ Extract the healthy green mask to use as the base plant area. """
    if hsv_bounds is None:
        hsv_bounds = {
            'lower': [35, 40, 40],
            'upper': [85, 255, 255]
        }
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_bounds['lower'], dtype=np.uint8)
    upper = np.array(hsv_bounds['upper'], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

def extract_pitting(patch_bgr, hsv_pitting_bounds=None):
    """
    Detect yellow/brown pitting.
    Typical yellow/brown in HSV: H from 10 to 35, S > 40, V > 40.
    """
    if hsv_pitting_bounds is None:
        hsv_pitting_bounds = {
            'lower': [10, 40, 40],
            'upper': [35, 255, 255]
        }
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_pitting_bounds['lower'], dtype=np.uint8)
    upper = np.array(hsv_pitting_bounds['upper'], dtype=np.uint8)
    
    pitting_mask = cv2.inRange(hsv, lower, upper)
    
    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pitting_mask = cv2.morphologyEx(pitting_mask, cv2.MORPH_OPEN, kernel)
    
    # Find pitting blobs
    contours, _ = cv2.findContours(pitting_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pitting_area = 0
    pitting_count = 0
    valid_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 5:  # Minimum 5 pixels to be considered pitting
            pitting_area += area
            pitting_count += 1
            valid_contours.append(cnt)
            
    return pitting_area, pitting_count, valid_contours

def extract_holes(patch_bgr, green_mask):
    """
    Detect internal shot-holes inside the leaf.
    Uses RETR_CCOMP to find internal contours (holes) within the green mask.
    """
    # Find contours with 2-level hierarchy
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    hole_area = 0
    hole_count = 0
    valid_hole_contours = []
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            # hierarchy[i] = [Next, Previous, First_Child, Parent]
            # If Parent != -1, it means this is an internal contour (a hole)
            if hierarchy[i][3] != -1:
                area = cv2.contourArea(cnt)
                if area >= 5: # Minimum 5 pixels to be considered a shot-hole
                    hole_area += area
                    hole_count += 1
                    valid_hole_contours.append(cnt)
                    
    return hole_area, hole_count, valid_hole_contours

def analyze_plant_biology(patch_bgr, hsv_bounds=None, hsv_pitting_bounds=None):
    """
    Aggregates all biological features for a single plant patch.
    """
    green_mask = get_green_mask(patch_bgr, hsv_bounds)
    
    # We need the actual pitting mask to avoid double-counting pitting as shot-holes.
    if hsv_pitting_bounds is None:
        hsv_pitting_bounds = {
            'lower': [10, 40, 40],
            'upper': [35, 255, 255]
        }
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_pitting_bounds['lower'], dtype=np.uint8)
    upper = np.array(hsv_pitting_bounds['upper'], dtype=np.uint8)
    pitting_mask = cv2.inRange(hsv, lower, upper)
    
    # Clean pitting mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pitting_mask = cv2.morphologyEx(pitting_mask, cv2.MORPH_OPEN, kernel)
    
    # Calculate pitting area and count
    contours, _ = cv2.findContours(pitting_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pitting_area = 0
    pitting_count = 0
    pitting_cnts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 5:
            pitting_area += area
            pitting_count += 1
            pitting_cnts.append(cnt)
            
    # For hole extraction, treat pitting as "leaf tissue" so it doesn't form an internal hole contour
    solid_leaf_mask = cv2.bitwise_or(green_mask, pitting_mask)
    hole_area, hole_count, hole_cnts = extract_holes(patch_bgr, solid_leaf_mask)
    
    plant_area = cv2.countNonZero(solid_leaf_mask)
    if plant_area == 0:
        plant_area = 1 # Prevent division by zero
    
    total_affected_area = pitting_area + hole_area
    pitting_to_hole_ratio = pitting_area / hole_area if hole_area > 0 else float('inf')
    if hole_area == 0 and pitting_area == 0:
        pitting_to_hole_ratio = 0.0
        
    metrics = {
        'plant_area': float(plant_area),
        'pitting_area': float(pitting_area),
        'pitting_count': int(pitting_count),
        'hole_area': float(hole_area),
        'hole_count': int(hole_count),
        'total_affected_area': float(total_affected_area),
        'pitting_to_hole_ratio': float(pitting_to_hole_ratio),
        'damage_percentage': float(total_affected_area / (plant_area + hole_area) * 100)
    }
    
    return metrics, pitting_cnts, hole_cnts

def draw_biology_overlay(patch_bgr, pitting_cnts, hole_cnts):
    """
    Draw red outlines around pitting and blue outlines around holes.
    """
    overlay = patch_bgr.copy()
    # Draw pitting (Red)
    cv2.drawContours(overlay, pitting_cnts, -1, (0, 0, 255), 1)
    # Draw holes (Blue)
    cv2.drawContours(overlay, hole_cnts, -1, (255, 0, 0), 1)
    return overlay
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Biological features extractor")
    parser.add_argument("--config", type=str, required=True, help="Path to audit config JSON")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    out_dir = Path(config["output_dir"])
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(config["manifest"])
    print(f"Loaded {len(df)} images for biological audit.")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['image_path']
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
            
        detection = detect_frame(bgr)
        if detection.status != "detected":
            continue
        frame, _ = crop_frame_interior(bgr, detection.corners)
        
        mask, regions = extract_plant_regions(frame)
        
        image_metrics = {
            'image_id': img_id,
            'num_plants': len(regions),
            'total_plant_area': 0,
            'total_pitting_area': 0,
            'total_hole_area': 0,
            'total_pitting_count': 0,
            'total_hole_count': 0
        }
        
        for r_idx, region in enumerate(regions):
            x, y, w, h = region.patch_box
            patch = frame[y:y+h, x:x+w]
            metrics, p_cnts, h_cnts = analyze_plant_biology(patch, config.get("hsv_bounds"), config.get("hsv_pitting_bounds"))
            
            image_metrics['total_plant_area'] += metrics['plant_area']
            image_metrics['total_pitting_area'] += metrics['pitting_area']
            image_metrics['total_hole_area'] += metrics['hole_area']
            image_metrics['total_pitting_count'] += metrics['pitting_count']
            image_metrics['total_hole_count'] += metrics['hole_count']
            
            overlay = draw_biology_overlay(patch, p_cnts, h_cnts)
            out_patch_path = overlays_dir / f"{img_id}_plant{r_idx}.jpg"
            cv2.imwrite(str(out_patch_path), overlay)
            
        results.append(image_metrics)
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "biology_audit_summary.csv", index=False)
    print(f"Saved audit summary to {out_dir / 'biology_audit_summary.csv'}")
