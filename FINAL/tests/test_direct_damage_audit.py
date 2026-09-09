import cv2
import numpy as np

from src.preprocessing.direct_damage_audit import (
    analyze_frame_damage,
    combined_damage_percentage,
)


def test_combined_damage_percentage():
    assert combined_damage_percentage(1000, 25, 75) == 10.0
    assert combined_damage_percentage(0, 25, 75) == 0.0


def test_frame_analysis_uses_xyxy_patch_coordinates():
    image = np.full((240, 320, 3), (70, 85, 105), dtype=np.uint8)
    cv2.ellipse(image, (160, 120), (55, 38), 0, 0, 360, (0, 255, 0), -1)
    cv2.circle(image, (145, 120), 7, (0, 0, 0), -1)
    cv2.circle(image, (175, 120), 7, (0, 255, 255), -1)
    metrics, leaf, holes, pitting = analyze_frame_damage(
        image,
        plant_parameters={
            "minimum_green_area": 10, "grouping_kernel_fraction": 0.02,
            "minimum_group_area": 20, "minimum_region_green_area": 20,
        },
    )
    assert metrics["region_count"] == 1
    assert metrics["leaf_area"] > 0
    assert metrics["hole_count"] == 1
    assert metrics["pitting_count"] == 1
    assert np.count_nonzero(leaf) == metrics["leaf_area"]
    assert np.count_nonzero(holes) > 0
    assert np.count_nonzero(pitting) > 0
