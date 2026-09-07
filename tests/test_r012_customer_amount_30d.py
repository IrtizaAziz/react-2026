"""Mandatory causal/parity preflight for the R012 30-day amount experiment."""
import inspect
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.customer_amount_30d import (FEATURES, WINDOW_NS, customer_amount_30d_chunked,
                                     customer_amount_30d_from_prior_stream,
                                     customer_amount_30d_production)
from src.data import REACT2026_CUSTOMER_AMOUNT_30D_FEATURES, load_training
from src.validation import make_splits


class R012CustomerAmount30DTests(unittest.TestCase):
    def _frame(self):
        day = int(WINDOW_NS // 30 // 1_000_000_000)
        return pd.DataFrame({
            "transaction_id": list("abcdef"),
            "timestamp": [0, 1, day, 30 * day, 30 * day, 30 * day + 1],
            "customer_id": ["x", "x", "x", "x", "y", "x"],
            "amount_bdt": [10., 20., 30., 40., 5., 50.],
            "customer_prior_mean_amount": [np.nan, 10., 15., 20., np.nan, 25.],
            "fraud": [0, 1, 0, 1, 0, 1],
        })

    def test_causal_window_semantics_and_no_label_api(self):
        base = self._frame()
        output = customer_amount_30d_production(base.timestamp, base.customer_id, base.amount_bdt,
                                                base.customer_prior_mean_amount)
        self.assertEqual(list(output), FEATURES)
        # Empty history; then a single prior transaction; then multiple priors.
        self.assertEqual(output.customer_prior_30d_count.iloc[0], 0)
        self.assertTrue(output.iloc[0, 1:].isna().all())
        self.assertEqual(output.customer_prior_30d_count.iloc[1], 1)
        self.assertEqual(output.customer_prior_30d_mean_amount.iloc[1], 10.)
        self.assertEqual(output.customer_prior_30d_count.iloc[2], 2)
        self.assertEqual(output.customer_prior_30d_mean_amount.iloc[2], 15.)
        # t - 30 days is retained; transactions at t do not enter any same-timestamp query.
        self.assertEqual(output.customer_prior_30d_count.iloc[3], 3)
        self.assertEqual(output.customer_prior_30d_count.iloc[4], 0)
        self.assertEqual(output.customer_prior_30d_count.iloc[5], 3)
        self.assertEqual(output.customer_prior_30d_mean_amount.iloc[5], 30.)
        self.assertAlmostEqual(output.customer_amount_to_prior_30d_mean.iloc[5], 50. / 31.)
        self.assertAlmostEqual(output.customer_30d_to_lifetime_mean_ratio.iloc[5], 31. / 26.)
        self.assertNotIn("fraud", inspect.signature(customer_amount_30d_production).parameters)
        self.assertNotIn("label", inspect.signature(customer_amount_30d_production).parameters)

    def test_permutation_future_chunk_and_continuation_parity(self):
        base = self._frame()
        output = customer_amount_30d_production(base.timestamp, base.customer_id, base.amount_bdt,
                                                base.customer_prior_mean_amount)
        # Reordering rows within the t=30d timestamp batch changes no per-ID features.
        permuted = pd.concat([base.iloc[:3], base.iloc[[4, 3]], base.iloc[5:]], ignore_index=True)
        shuffled = customer_amount_30d_production(permuted.timestamp, permuted.customer_id, permuted.amount_bdt,
                                                  permuted.customer_prior_mean_amount)
        pd.testing.assert_frame_equal(pd.concat([base.transaction_id, output], axis=1).set_index("transaction_id").sort_index(),
                                      pd.concat([permuted.transaction_id, shuffled], axis=1).set_index("transaction_id").sort_index())
        tied = pd.DataFrame({"timestamp": [0, 10, 10], "customer_id": ["x", "x", "x"],
                             "amount_bdt": [10., 20., 30.], "customer_prior_mean_amount": [np.nan, 10., 10.]})
        tied_output = customer_amount_30d_production(tied.timestamp, tied.customer_id, tied.amount_bdt,
                                                     tied.customer_prior_mean_amount)
        self.assertEqual(tied_output.customer_prior_30d_count.iloc[1], 1)
        self.assertEqual(tied_output.customer_prior_30d_count.iloc[2], 1)
        # Appended or mutated future amounts cannot change earlier features.
        future = pd.concat([base, pd.DataFrame({"transaction_id": ["z"], "timestamp": [4_000_000], "customer_id": ["x"], "amount_bdt": [999.], "customer_prior_mean_amount": [30.], "fraud": [0]})], ignore_index=True)
        future_output = customer_amount_30d_production(future.timestamp, future.customer_id, future.amount_bdt,
                                                       future.customer_prior_mean_amount)
        pd.testing.assert_frame_equal(output, future_output.iloc[:len(base)].reset_index(drop=True))
        chunks = [base.iloc[:2], base.iloc[2:4], base.iloc[4:]]
        pd.testing.assert_frame_equal(output, customer_amount_30d_chunked(chunks))
        continued = customer_amount_30d_from_prior_stream(base.iloc[:3].timestamp, base.iloc[:3].customer_id,
            base.iloc[:3].amount_bdt, base.iloc[:3].customer_prior_mean_amount, base.iloc[3:].timestamp,
            base.iloc[3:].customer_id, base.iloc[3:].amount_bdt, base.iloc[3:].customer_prior_mean_amount)
        pd.testing.assert_frame_equal(output.iloc[3:].reset_index(drop=True), continued)

    def test_r007_parity_manifest_and_saved_split(self):
        root = Path(ROOT)
        parent, candidate = load_config(root / "config_r007.json"), load_config(root / "config_r012.json")
        candidate.validate()
        self.assertEqual(candidate.features[:len(parent.features)], parent.features)
        self.assertEqual(candidate.features[len(parent.features):], REACT2026_CUSTOMER_AMOUNT_30D_FEATURES)
        self.assertEqual(len(candidate.features), 47)
        self.assertEqual(candidate.model_params, parent.model_params)
        candidate_frame, fingerprint = load_training(candidate, root)
        parent_frame, _ = load_training(parent, root)
        pd.testing.assert_frame_equal(candidate_frame[parent.features], parent_frame[parent.features],
                                      check_dtype=True, check_exact=True)
        _, _, splits = make_splits(candidate_frame, candidate, fingerprint,
                                   reuse=root / candidate.splits_file)
        self.assertEqual(splits["signature"], "6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47")


if __name__ == "__main__":
    unittest.main()
