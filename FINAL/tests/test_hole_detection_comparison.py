import cv2
import numpy as np
import pandas as pd
import json
import pytest

from src.preprocessing.biological_features import analyze_plant_biology
from src.preprocessing.hole_detection_comparison import summarize_review


def test_soil_similarity_recognizes_bright_soil_visible_through_hole():
    soil = (70, 85, 105)
    image = np.full((120, 140, 3), soil, dtype=np.uint8)
    cv2.ellipse(image, (70, 60), (48, 35), 0, 0, 360, (0, 255, 0), -1)
    cv2.circle(image, (55, 60), 7, soil, -1)
    cv2.circle(image, (85, 60), 7, (0, 255, 255), -1)
    old, *_ = analyze_plant_biology(image, hole_classification="brightness")
    new, *_ = analyze_plant_biology(
        image, hole_classification="soil_similarity", soil_lab_distance=35
    )
    assert old["hole_count"] == 0
    assert new["hole_count"] == 1
    assert new["pitting_count"] == 1


def test_comparison_summary_requires_manual_rows(tmp_path):
    pd.DataFrame({
        "review_status": [""], "pitting_still_usable": [""],
        "soil_false_holes": [""],
    }).to_csv(tmp_path / "hole_detection_comparison.csv", index=False)
    (tmp_path / "hole_detection_comparison_summary.json").write_text(
        json.dumps({"manual_review_complete": False}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        summarize_review(tmp_path)
