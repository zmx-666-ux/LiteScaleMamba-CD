import unittest

import numpy as np

from utils.metrics import BinaryMetrics


class BinaryMetricsTests(unittest.TestCase):
    def test_ignored_pixels_do_not_change_binary_scores(self):
        labels = np.array([[0, 1, 255], [1, 0, 0]], dtype=np.uint8)
        predictions = np.array([[0, 0, 1], [1, 1, 0]], dtype=np.uint8)
        meter = BinaryMetrics()
        meter.update(labels, predictions)
        scores = meter.compute()

        self.assertAlmostEqual(scores["precision"], 0.5)
        self.assertAlmostEqual(scores["recall"], 0.5)
        self.assertAlmostEqual(scores["f1"], 0.5)
        self.assertAlmostEqual(scores["iou"], 1 / 3)
        self.assertAlmostEqual(scores["oa"], 0.6)
        self.assertAlmostEqual(scores["kappa"], 1 / 6)

    def test_batches_are_aggregated_before_computing_f1(self):
        meter = BinaryMetrics()
        meter.update(np.array([1, 1]), np.array([1, 0]))
        meter.update(np.array([1, 0, 0]), np.array([1, 1, 0]))

        self.assertAlmostEqual(meter.compute()["f1"], 2 / 3)

    def test_invalid_predictions_are_rejected(self):
        meter = BinaryMetrics()
        with self.assertRaises(ValueError):
            meter.update(np.array([0]), np.array([2]))


if __name__ == "__main__":
    unittest.main()
