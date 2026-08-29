import unittest

from src.evaluation.diagnose import summarize_predictions


class TestDiagnostics(unittest.TestCase):
    def test_reports_spread_and_constant_baselines(self):
        result = summarize_predictions([0, 10, 20], [2, 8, 12], {"train_median": 10})
        self.assertAlmostEqual(result["mae"], 4.0)
        self.assertAlmostEqual(result["predictions"]["minimum"], 2.0)
        self.assertAlmostEqual(result["predictions"]["maximum"], 12.0)
        self.assertAlmostEqual(result["constant_baselines"]["train_median"]["mae"], 20 / 3)


if __name__ == "__main__":
    unittest.main()
