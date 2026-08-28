import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.baselines import evaluate_constant_baselines


class TestConstantBaselines(unittest.TestCase):
    def test_real_split_baselines(self):
        manifest = Path("outputs/tables/baseline_manifest_split.csv")
        if not manifest.is_file():
            self.skipTest("Build baseline splits first")
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "metrics.json"
            results = evaluate_constant_baselines(manifest, output)
            self.assertEqual(results["train_samples"], 331)
            self.assertEqual(results["test_samples"], 73)
            self.assertGreater(results["mean_predictor"]["mae"], 0)
            self.assertEqual(json.loads(output.read_text())["test_samples"], 73)


if __name__ == "__main__":
    unittest.main()
