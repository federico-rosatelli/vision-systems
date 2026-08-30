import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.plant_regions import (
    approve_plant_region_audit,
    extract_plant_regions,
    vegetation_mask,
)


class TestPlantRegions(unittest.TestCase):
    def test_green_regions_are_detected_and_brown_soil_is_rejected(self):
        image = np.full((500, 600, 3), (70, 85, 105), dtype=np.uint8)
        cv2.circle(image, (150, 180), 34, (55, 180, 75), -1)
        cv2.circle(image, (205, 185), 32, (60, 170, 80), -1)
        cv2.circle(image, (450, 330), 42, (65, 190, 85), -1)
        mask, regions = extract_plant_regions(
            image, minimum_green_area=20, grouping_kernel_fraction=0.04,
            minimum_group_area=100, minimum_region_green_area=100,
            patch_padding_fraction=0.02,
        )
        self.assertEqual(mask[180, 150], 255)
        self.assertEqual(mask[20, 20], 0)
        self.assertEqual(len(regions), 2)
        self.assertTrue(all(region.green_area > 1000 for region in regions))

    def test_tiny_noise_is_removed(self):
        image = np.full((200, 200, 3), (70, 85, 105), dtype=np.uint8)
        image[20:22, 20:22] = (30, 220, 30)
        mask = vegetation_mask(image, minimum_green_area=20)
        self.assertEqual(np.count_nonzero(mask), 0)

    def test_manual_approval_updates_audit_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            pd.DataFrame({
                "automatic_status": ["detected", "detected"],
                "manual_status": ["", ""], "manual_notes": ["", ""],
            }).to_csv(output / "plant_region_audit.csv", index=False)
            (output / "plant_region_audit_summary.json").write_text(
                '{"manual_review_complete": false}\n', encoding="utf-8"
            )
            summary = approve_plant_region_audit(output, reviewer="tester")
            audit = pd.read_csv(output / "plant_region_audit.csv")
        self.assertTrue(summary["manual_review_complete"])
        self.assertEqual(set(audit["manual_status"]), {"approved"})


if __name__ == "__main__":
    unittest.main()
