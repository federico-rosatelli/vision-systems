import unittest

from src.evaluation.evaluate import pairwise_ranking_accuracy


class TestEvaluationMetrics(unittest.TestCase):
    def test_pairwise_ranking_accuracy_respects_gap(self):
        result = pairwise_ranking_accuracy(
            [0.0, 4.0, 10.0], [1.0, 4.0, 9.0], minimum_gap=5.0
        )
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
