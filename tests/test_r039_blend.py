import unittest
import numpy as np
from src.r039_blend import WEIGHTS, blend_covered_probabilities


class R039CoverageTests(unittest.TestCase):
    def test_identical_expected_null_masks_are_accepted_and_preserved(self):
        a = np.array([[np.nan, np.nan], [.8, .2], [.1, .9]])
        b = np.array([[np.nan, np.nan], [.6, .4], [.3, .7]])
        out, mask = blend_covered_probabilities(a, b)
        self.assertTrue(np.array_equal(mask, [False, True, True]))
        self.assertTrue(np.isnan(out[0]).all())
        self.assertTrue(np.allclose(out[1], [.75, .25]))

    def test_mismatched_coverage_and_invalid_probability_fail_closed(self):
        with self.assertRaises(ValueError): blend_covered_probabilities([[np.nan, np.nan], [.5, .5]], [[.5, .5], [.5, .5]])
        with self.assertRaises(ValueError): blend_covered_probabilities([[.2, .8]], [[1.2, -.2]])

    def test_weights_are_exactly_locked(self):
        self.assertEqual(WEIGHTS, (.75, .25))
        with self.assertRaises(ValueError): blend_covered_probabilities([[.5, .5]], [[.5, .5]], (.5, .5))

    def test_canonical_masks_drive_metrics_not_blanket_notnull(self):
        # The production runner scores each established fold/window mask explicitly;
        # uncovered warmup rows cannot enter any canonical temporal mask.
        out, mask = blend_covered_probabilities([[np.nan, np.nan], [.9, .1]], [[np.nan, np.nan], [.8, .2]])
        self.assertFalse(mask[0]); self.assertTrue(np.isnan(out[0]).all())

    def test_id_and_split_mismatches_are_contract_failures(self):
        # Integration checks in run() require equal __row__, transaction_id, __fold__, and split signature.
        a = np.array([[.5, .5]])
        self.assertTrue(blend_covered_probabilities(a, a)[1][0])


if __name__ == '__main__': unittest.main()
