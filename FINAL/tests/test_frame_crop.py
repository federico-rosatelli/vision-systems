import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.frame_crop import (
    approve_frame_audit,
    crop_frame_interior,
    detect_frame,
    select_audit_rows,
)


class TestFrameCrop(unittest.TestCase):
    def test_detects_synthetic_metal_frame(self):
        image = np.full((750, 1000, 3), (75, 95, 110), dtype=np.uint8)
        color = (190, 185, 170)
        cv2.rectangle(image, (300, 150), (700, 600), color, 14)
        cv2.line(image, (500, 150), (500, 600), color, 10)
        cv2.line(image, (300, 375), (700, 375), color, 10)
        detection = detect_frame(image, working_width=1000)
        self.assertEqual(detection.status, "detected", detection.reason)
        self.assertGreater(detection.confidence, 0.5)
        np.testing.assert_allclose(
            detection.corners,
            np.array([[300, 150], [700, 150], [700, 600], [300, 600]]),
            atol=20,
        )
        crop, _ = crop_frame_interior(image, detection.corners)
        self.assertGreater(crop.shape[0], 400)
        self.assertGreater(crop.shape[1], 350)

    def test_audit_selection_excludes_test_and_covers_strata(self):
        rows = []
        for split in ("train", "val", "test"):
            for score in (1.0, 5.0, 10.0, 20.0):
                for index in range(5):
                    rows.append({
                        "filename": f"{split}_{score}_{index}.jpg",
                        "image_path": "/unused.jpg", "split": split,
                        "mean_score": score,
                    })
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            selected = select_audit_rows(manifest, sample_size=30, seed=42)
        self.assertEqual(len(selected), 30)
        self.assertEqual(set(selected["split"]), {"train", "val"})
        self.assertEqual(selected["split"].value_counts().to_dict(), {"train": 15, "val": 15})
        self.assertEqual(set(selected["score_bin"].astype(str)), {
            "low_0_3", "medium_3_8", "high_8_15", "very_high_15_plus",
        })

    def test_manual_approval_updates_audit_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            pd.DataFrame({
                "automatic_status": ["detected", "detected"],
                "manual_status": ["", ""], "manual_notes": ["", ""],
            }).to_csv(output / "frame_audit.csv", index=False)
            (output / "frame_audit_summary.json").write_text(
                '{"manual_review_complete": false}\n', encoding="utf-8"
            )
            summary = approve_frame_audit(output, reviewer="tester")
            audit = pd.read_csv(output / "frame_audit.csv")
        self.assertTrue(summary["manual_review_complete"])
        self.assertEqual(summary["reviewer"], "tester")
        self.assertEqual(set(audit["manual_status"]), {"approved"})


if __name__ == "__main__":
    unittest.main()
