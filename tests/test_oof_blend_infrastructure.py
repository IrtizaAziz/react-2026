import unittest
import numpy as np
import pandas as pd

from src.config import Config
from src.ensemble import blend_covered_oofs, score_canonical_oof


def item(values, folds=(-1, 0, 1)):
    frame = pd.DataFrame({"__row__": range(3), "transaction_id": pd.Series(["a", "b", "c"], dtype="string"), "__fold__": folds})
    metadata = {"identity_hash": "ids", "data_fingerprint": "data", "class_order": [0, 1],
                "prediction_kind": "probability", "prediction_columns": ["pred_0", "pred_1"],
                "id_columns": ["transaction_id"], "split_signature": "split"}
    return frame, np.asarray(values, dtype=float), metadata


class CoveredOofBlendTests(unittest.TestCase):
    def test_r039_regression_expected_warmup_nulls_are_preserved(self):
        # R039 failed after coverage handling because generic blending previously rejected its valid warmup nulls.
        first = item([[np.nan, np.nan], [.8, .2], [.1, .9]])
        second = item([[np.nan, np.nan], [.6, .4], [.3, .7]])
        frame, blended, _meta, covered = blend_covered_oofs([first, second], [.75, .25])
        self.assertTrue(np.array_equal(covered, [False, True, True]))
        self.assertTrue(np.isnan(blended[0]).all())
        self.assertTrue(np.allclose(blended[1], [.75, .25]))
        self.assertEqual(frame.loc[covered, "__fold__"].tolist(), [0, 1])

    def test_mismatched_or_partial_null_masks_fail_before_blending(self):
        first = item([[np.nan, np.nan], [.8, .2], [.1, .9]])
        mismatch = item([[.5, .5], [.6, .4], [.3, .7]])
        partial = item([[np.nan, np.nan], [.6, np.nan], [.3, .7]])
        with self.assertRaisesRegex(ValueError, "coverage"):
            blend_covered_oofs([first, mismatch], [.75, .25])
        with self.assertRaisesRegex(ValueError, "entirely"):
            blend_covered_oofs([first, partial], [.75, .25])

    def test_r039_reporting_regression_scores_canonical_windows_not_legacy_diagnostic_names(self):
        # R039 raised KeyError('R017') by looking for its newer diagnostic labels in R017's old report.
        frame = pd.DataFrame({"fraud": [0, 0, 1, 0, 1], "timestamp": ["2026-04-01", "2026-05-20", "2026-06-01", "2026-06-20", "2026-07-10"]})
        probabilities = np.array([[np.nan, np.nan], [.8, .2], [.1, .9], [.8, .2], [.1, .9]])
        config = Config(target="fraud", time_column="timestamp", n_splits=2, task="classification",
                        metric="average_precision", metric_direction="higher", prediction_kind="probability",
                        class_order=[0, 1], positive_class=1)
        scores = score_canonical_oof(frame, probabilities, [-1, 0, 0, 1, 1], config,
                                     [("F2_late_corrected", 1, "2026-06-15", "2026-07-16")])
        self.assertIn("F2_late_corrected", scores)
        self.assertIn("F2", scores)


if __name__ == "__main__":
    unittest.main()
