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

def analyze_plant_biology(patch_bgr, hsv_bounds=None, hsv_pitting_bounds=None):
    """
    Aggregates all biological features for a single plant patch using a topological approach.
    It fills the external contours of the green leaf to find the solid leaf area,
    and isolates internal non-green blobs as damage. Dark blobs are holes, bright blobs are pitting.
    """
    green_mask = get_green_mask(patch_bgr, hsv_bounds)
    
    # 1. Create filled leaf mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_leaf_mask = np.zeros_like(green_mask)
    cv2.drawContours(filled_leaf_mask, contours, -1, 255, -1)
    
    # 2. Isolate internal damage (holes and pitting)
    damage_mask = cv2.bitwise_and(filled_leaf_mask, cv2.bitwise_not(green_mask))
    
    # Clean up small noise in damage mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN, kernel)
    
    # 3. Find damage blobs
    damage_contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    
    pitting_area = 0
    pitting_count = 0
    pitting_cnts = []
    
    hole_area = 0
    hole_count = 0
    hole_cnts = []
    
    for cnt in damage_contours:
        area = cv2.contourArea(cnt)
        if area < 5:
            continue
            
        # Create a mask for this specific blob to analyze its color
        blob_mask = np.zeros_like(green_mask)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
        
        # Calculate mean Value (brightness) of the blob
        mean_v = cv2.mean(hsv[:, :, 2], mask=blob_mask)[0]
        
        # If the blob is bright, it's necrosis/pitting. If it's dark (shadow/soil), it's a hole.
        # Soil and shadows are typically V < 40. Necrotic tissue is usually brighter.
        if mean_v >= 40:
            pitting_area += area
            pitting_count += 1
            pitting_cnts.append(cnt)
        else:
            hole_area += area
            hole_count += 1
            hole_cnts.append(cnt)
            
    plant_area = cv2.countNonZero(filled_leaf_mask)
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
        'damage_percentage': float(total_affected_area / plant_area * 100)
    }
    
    return metrics, pitting_cnts, hole_cnts, filled_leaf_mask

def draw_biology_overlay(patch_bgr, pitting_cnts, hole_cnts, solid_leaf_mask=None):
    """
    Draw red outlines around pitting and blue outlines around holes.
    Optionally masks out the background (soil) to black using solid_leaf_mask.
    """
    overlay = patch_bgr.copy()
    
    if solid_leaf_mask is not None:
        overlay = cv2.bitwise_and(overlay, overlay, mask=solid_leaf_mask)
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
            metrics, p_cnts, h_cnts, solid_mask = analyze_plant_biology(patch, config.get("hsv_bounds"), config.get("hsv_pitting_bounds"))
            
            image_metrics['total_plant_area'] += metrics['plant_area']
            image_metrics['total_pitting_area'] += metrics['pitting_area']
            image_metrics['total_hole_area'] += metrics['hole_area']
            image_metrics['total_pitting_count'] += metrics['pitting_count']
            image_metrics['total_hole_count'] += metrics['hole_count']
            
            overlay = draw_biology_overlay(patch, p_cnts, h_cnts, solid_mask)
            out_patch_path = overlays_dir / f"{img_id}_plant{r_idx}.jpg"
            cv2.imwrite(str(out_patch_path), overlay)
            
        results.append(image_metrics)
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "biology_audit_summary.csv", index=False)
    print(f"Saved audit summary to {out_dir / 'biology_audit_summary.csv'}")
