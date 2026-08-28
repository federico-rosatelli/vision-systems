import json
import tempfile
import unittest
from pathlib import Path

from src.training.provenance import build_run_metadata, save_json, sha256_file


class TestRunProvenance(unittest.TestCase):
    def test_metadata_is_complete_and_serializable(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifest = root / "manifest.csv"
            manifest.write_text("filename,split\na.jpg,train\n", encoding="utf-8")
            metadata = build_run_metadata(
                Path.cwd(), manifest, "test_run",
                {"model_name": "dinov3_vits16", "image_size": 224},
                {"seed": 42, "training_mode": "regression"},
            )
            output = root / "run_config.json"
            save_json(metadata, output)
            saved = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(saved["manifest_sha256"], sha256_file(manifest))
            self.assertEqual(saved["run_name"], "test_run")
            self.assertEqual(saved["model"]["model_name"], "dinov3_vits16")
            self.assertIn("git_commit", saved)
            self.assertIn("dependencies", saved)


if __name__ == "__main__":
    unittest.main()
