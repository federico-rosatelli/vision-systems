import cv2
import numpy as np
import pytest

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.biological_features import get_green_mask, extract_pitting, extract_holes, analyze_plant_biology

def create_synthetic_plant(width=100, height=100):
    """
    Creates a synthetic BGR image representing a green leaf with:
    - 1 shot hole (black circle inside the green leaf)
    - 1 pitting spot (yellow circle inside the green leaf)
    """
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw green leaf (ellipse)
    # Green in BGR is (0, 255, 0)
    cv2.ellipse(bgr, (50, 50), (40, 30), 0, 0, 360, (0, 255, 0), -1)
    
    # Draw hole (black circle, radius 5)
    cv2.circle(bgr, (35, 50), 5, (0, 0, 0), -1)
    
    # Draw pitting (yellow circle, radius 5)
    # Yellow in BGR is (0, 255, 255)
    cv2.circle(bgr, (65, 50), 5, (0, 255, 255), -1)
    
    return bgr

def test_extract_pitting():
    bgr = create_synthetic_plant()
    
    # The default pitting bounds are H=[10, 35], S=[40, 255], V=[40, 255].
    # Pure yellow is BGR(0, 255, 255) -> HSV(30, 255, 255), which falls in the bounds.
    pitting_area, pitting_count, cnts = extract_pitting(bgr)
    
    assert pitting_count == 1
    # Area of circle with radius 5 is approx pi * 25 ~ 78.5
    # Pixels might vary slightly, so check within range
    assert 60 < pitting_area < 90

def test_analyze_plant_biology():
    bgr = create_synthetic_plant()
    
    metrics, p_cnts, h_cnts = analyze_plant_biology(bgr)
    
    assert metrics['pitting_count'] == 1
    assert metrics['hole_count'] == 1
    assert metrics['plant_area'] > 2000
    assert metrics['total_affected_area'] == metrics['pitting_area'] + metrics['hole_area']
    assert metrics['damage_percentage'] > 0
    assert metrics['pitting_to_hole_ratio'] > 0
