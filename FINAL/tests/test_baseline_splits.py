import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data.make_baseline_splits import create_baseline_splits


class TestBaselineSplits(unittest.TestCase):
    def test_real_baseline_split_contract(self):
        manifest = Path("outputs/tables/baseline_manifest.csv")
        if not manifest.is_file():
            self.skipTest("Build the baseline manifest first")

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            split_manifest = root / "split.csv"
            audit_path = root / "audit.json"
            groups_path = root / "groups.json"
            audit = create_baseline_splits(
                manifest, split_manifest, audit_path, groups_path, seed=42
            )

            with split_manifest.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            with groups_path.open(encoding="utf-8") as handle:
                groups = json.load(handle)

            self.assertEqual(len(rows), 470)
            self.assertEqual({row["split"] for row in rows}, {"train", "val", "test"})
            self.assertEqual(len({row["filename"] for row in rows}), 470)
            self.assertFalse(set(groups["train"]) & set(groups["val"]))
            self.assertFalse(set(groups["train"]) & set(groups["test"]))
            self.assertFalse(set(groups["val"]) & set(groups["test"]))
            for plot_group in {row["plot_group"] for row in rows}:
                self.assertEqual(len({row["split"] for row in rows if row["plot_group"] == plot_group}), 1)
            self.assertEqual(audit["status"], "pass")
            for split, target in {"train": 0.70, "val": 0.15, "test": 0.15}.items():
                self.assertLessEqual(abs(audit["image_ratios"][split] - target), 0.03)


if __name__ == "__main__":
    unittest.main()
