"""Mandatory causal/parity preflight for R013 customer--merchant familiarity."""
import inspect
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.customer_merchant import FEATURES, customer_merchant_chunked, customer_merchant_production
from src.data import REACT2026_CUSTOMER_MERCHANT_FEATURES, load_training
from src.validation import make_splits


class R013CustomerMerchantTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame({"transaction_id": list("abcdef"), "timestamp": [0, 1, 2, 2, 3, 4],
                             "customer_id": ["x", "x", "x", "x", "x", "y"],
                             "merchant_id": ["m1", "m1", "m2", "m1", "m1", "m1"],
                             "customer_prior_count": [0, 1, 2, 2, 4, 0], "fraud": [0, 1, 0, 1, 0, 1]})

    def test_pair_counts_recency_share_and_label_isolation(self):
        base = self._frame()
        output = customer_merchant_production(base.timestamp, base.customer_id, base.merchant_id, base.customer_prior_count)
        self.assertEqual(list(output), FEATURES)
        self.assertEqual(output.customer_merchant_prior_count.tolist(), [0, 1, 0, 2, 3, 0])
        self.assertEqual(output.customer_merchant_is_new.tolist(), [1, 0, 1, 0, 0, 1])
        self.assertTrue(np.isnan(output.customer_merchant_seconds_since_last.iloc[0]))
        self.assertEqual(output.customer_merchant_seconds_since_last.iloc[1], 1.)
        self.assertEqual(output.customer_merchant_seconds_since_last.iloc[4], 1.)
        self.assertEqual(output.customer_merchant_share_of_customer_history.tolist()[:5], [0., 1., 0., 1., .75])
        self.assertNotIn("fraud", inspect.signature(customer_merchant_production).parameters)
        self.assertNotIn("label", inspect.signature(customer_merchant_production).parameters)

    def test_tied_permutation_future_and_chunk_parity(self):
        base = self._frame()
        output = customer_merchant_production(base.timestamp, base.customer_id, base.merchant_id, base.customer_prior_count)
        # Both x/m1 rows at timestamp 2 query the identical pre-t pair state.
        tied = pd.DataFrame({"timestamp": [0, 10, 10], "customer_id": ["x", "x", "x"],
                             "merchant_id": ["m", "m", "m"], "customer_prior_count": [0, 1, 1]})
        tied_output = customer_merchant_production(tied.timestamp, tied.customer_id, tied.merchant_id, tied.customer_prior_count)
        self.assertEqual(tied_output.customer_merchant_prior_count.iloc[1], 1)
        self.assertEqual(tied_output.customer_merchant_prior_count.iloc[2], 1)
        permuted = pd.concat([base.iloc[:2], base.iloc[[3, 2]], base.iloc[4:]], ignore_index=True)
        shuffled = customer_merchant_production(permuted.timestamp, permuted.customer_id, permuted.merchant_id, permuted.customer_prior_count)
        pd.testing.assert_frame_equal(pd.concat([base.transaction_id, output], axis=1).set_index("transaction_id").sort_index(),
                                      pd.concat([permuted.transaction_id, shuffled], axis=1).set_index("transaction_id").sort_index())
        future = pd.concat([base, pd.DataFrame({"transaction_id":["z"],"timestamp":[99],"customer_id":["x"],"merchant_id":["changed"],"customer_prior_count":[5],"fraud":[0]})], ignore_index=True)
        future_output = customer_merchant_production(future.timestamp, future.customer_id, future.merchant_id, future.customer_prior_count)
        pd.testing.assert_frame_equal(output, future_output.iloc[:len(base)].reset_index(drop=True))
        pd.testing.assert_frame_equal(output, customer_merchant_chunked([base.iloc[:2], base.iloc[2:4], base.iloc[4:]]))

    def test_r007_parity_manifest_and_saved_split(self):
        root = Path(ROOT); parent, candidate = load_config(root / "config_r007.json"), load_config(root / "config_r013.json")
        candidate.validate()
        self.assertEqual(candidate.features[:len(parent.features)], parent.features)
        self.assertEqual(candidate.features[len(parent.features):], REACT2026_CUSTOMER_MERCHANT_FEATURES)
        self.assertEqual(len(candidate.features), 45); self.assertEqual(candidate.model_params, parent.model_params)
        candidate_frame, fingerprint = load_training(candidate, root); parent_frame, _ = load_training(parent, root)
        pd.testing.assert_frame_equal(candidate_frame[parent.features], parent_frame[parent.features], check_dtype=True, check_exact=True)
        _, _, splits = make_splits(candidate_frame, candidate, fingerprint, reuse=root / candidate.splits_file)
        self.assertEqual(splits["signature"], "6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47")


if __name__ == "__main__":
    unittest.main()
