"""Regression coverage for R018's F2 screen defect."""
import unittest
import numpy as np
import pandas as pd
from src.config import Config
from src.train import early_screen_scores


class R019EarlyScreenTests(unittest.TestCase):
    def test_f2_means_complete_fold_and_subwindows_remain_half_open(self):
        frame = pd.DataFrame({"timestamp": ["2026-05-15", "2026-06-14", "2026-06-15", "2026-07-01", "2026-07-15"], "fraud": [0, 1, 0, 1, 1]})
        config = Config(target="fraud", time_column="timestamp", metric="average_precision", metric_direction="higher", task="classification", prediction_kind="probability", class_order=[0, 1], positive_class=1, n_splits=2, early_stop_screen={"fold": 1, "references": {"F2": 0.0, "F2_late_half": 0.0, "July_1_15": 0.0}, "maximum_drops": {"F2": 1.0, "F2_late_half": 1.0, "July_1_15": 1.0}}, diagnostic_windows=[{"name": "F2_late_half", "fold": 1, "start": "2026-06-15", "end": "2026-07-16"}, {"name": "July_1_15", "fold": 1, "start": "2026-07-01", "end": "2026-07-16"}])
        predictions = np.array([[.9, .1], [.2, .8], [.7, .3], [.3, .7], [.1, .9]])
        got = early_screen_scores(frame, predictions, np.ones(len(frame), dtype=int), config, 1)
        self.assertAlmostEqual(got["F2"], 1.0)
        self.assertAlmostEqual(got["F2_late_half"], 1.0)
        self.assertAlmostEqual(got["July_1_15"], 1.0)


if __name__ == "__main__":
    unittest.main()
