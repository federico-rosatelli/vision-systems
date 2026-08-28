import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data.make_baseline_manifest import build_baseline_manifest


class TestBaselineManifest(unittest.TestCase):
    def test_real_read_only_dataset_contract(self):
        raw_dir = Path("/home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage")
        if not raw_dir.is_dir():
            self.skipTest("Shared read-only dataset is unavailable")

        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest_path = Path(temporary_dir) / "baseline_manifest.csv"
            audit_path = Path(temporary_dir) / "baseline_audit.json"
            audit = build_baseline_manifest(raw_dir, manifest_path, audit_path)

            with manifest_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            with audit_path.open(encoding="utf-8") as handle:
                saved_audit = json.load(handle)

            self.assertEqual(len(rows), 470)
            self.assertEqual(len({row["filename"] for row in rows}), 470)
            self.assertEqual(len({row["image_path"] for row in rows}), 470)
            self.assertTrue(all(Path(row["image_path"]).is_file() for row in rows))
            self.assertTrue(all(row["plot_group"] for row in rows))
            self.assertTrue(all(row["genotype"] for row in rows))
            self.assertTrue(all(float(row["disagreement"]) <= 5.0 for row in rows))
            self.assertEqual(audit["disagreement_lt_5"], 443)
            self.assertEqual(audit["disagreement_eq_5"], 27)
            self.assertEqual(saved_audit["status"], "pass")


if __name__ == "__main__":
    unittest.main()
