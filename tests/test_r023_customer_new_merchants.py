"""Mandatory causal, parity, and locked-split preflight for R023."""
import inspect
from pathlib import Path
import unittest
import pandas as pd

from src.config import ROOT, load_config
from src.customer_new_merchants import (FEATURES, customer_new_merchants_chunked,
                                        customer_new_merchants_production,
                                        customer_new_merchants_simple_oracle)
from src.data import REACT2026_CUSTOMER_NEW_MERCHANTS_FEATURES, load_training
from src.validation import make_splits


class R023CustomerNewMerchantTests(unittest.TestCase):
    def _frame(self):
        day = 86_400
        return pd.DataFrame({"transaction_id": list("abcdefghij"),
            "timestamp": [0, 0, day, day, 2*day, 8*day, 8*day, 31*day, 32*day, 32*day],
            "customer_id": ["a", "a", "a", "a", "a", "a", "b", "a", "a", "a"],
            "merchant_id": ["m", "m", "m", "n", "o", "m", "x", "p", "q", "q"],
            "fraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

    def test_strict_past_ties_duplicates_oracle_and_independence(self):
        base = self._frame(); actual = customer_new_merchants_production(base.timestamp, base.customer_id, base.merchant_id)
        self.assertEqual(list(actual), FEATURES)
        self.assertEqual(actual.customer_new_merchants_7d.tolist()[:5], [0, 0, 1, 1, 2])
        self.assertEqual(actual.customer_new_merchants_30d.tolist()[:5], [0, 0, 1, 1, 2])
        self.assertAlmostEqual(actual.customer_new_merchant_share_7d.iloc[2], .5)
        self.assertEqual(actual.customer_new_merchants_7d.iloc[0], actual.customer_new_merchants_7d.iloc[1])
        self.assertEqual(actual.customer_new_merchants_7d.iloc[8], actual.customer_new_merchants_7d.iloc[9])
        # Duplicate same-timestamp (a,m) rows create one relationship event, while both remain transaction denominators.
        self.assertEqual(actual.customer_new_merchants_7d.iloc[2], 1); self.assertAlmostEqual(actual.customer_new_merchant_share_7d.iloc[2], .5)
        self.assertNotIn("fraud", inspect.signature(customer_new_merchants_production).parameters)
        pd.testing.assert_frame_equal(actual, customer_new_merchants_simple_oracle(base.timestamp, base.customer_id, base.merchant_id))
        permuted = pd.concat([base.iloc[:2], base.iloc[[3,2]], base.iloc[4:]], ignore_index=True)
        got = customer_new_merchants_production(permuted.timestamp, permuted.customer_id, permuted.merchant_id)
        pd.testing.assert_frame_equal(pd.concat([base.transaction_id, actual], axis=1).set_index("transaction_id").sort_index(), pd.concat([permuted.transaction_id, got], axis=1).set_index("transaction_id").sort_index())
        future = pd.concat([base, pd.DataFrame({"transaction_id":["z"], "timestamp":[99*86_400], "customer_id":["future_customer"], "merchant_id":["future_merchant"], "fraud":[1]})], ignore_index=True)
        future_actual = customer_new_merchants_production(future.timestamp, future.customer_id, future.merchant_id).iloc[:len(base)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, future_actual)
        mutated = future.copy(); mutated.loc[len(base), ["customer_id", "merchant_id"]] = ["a", "m"]
        pd.testing.assert_frame_equal(actual, customer_new_merchants_production(mutated.timestamp, mutated.customer_id, mutated.merchant_id).iloc[:len(base)].reset_index(drop=True))
        labels = future.copy(); labels["fraud"] = 1 - labels["fraud"]
        pd.testing.assert_frame_equal(customer_new_merchants_production(future.timestamp, future.customer_id, future.merchant_id), customer_new_merchants_production(labels.timestamp, labels.customer_id, labels.merchant_id))
        pd.testing.assert_frame_equal(actual, customer_new_merchants_chunked([base.iloc[:2], base.iloc[2:6], base.iloc[6:]]))

    def test_r017_parity_row_ids_and_locked_split(self):
        root = Path(ROOT); parent, candidate = load_config(root / "config_r017.json"), load_config(root / "config_r023.json"); candidate.validate()
        self.assertEqual(candidate.features[:len(parent.features)], parent.features)
        self.assertEqual(candidate.features[len(parent.features):], REACT2026_CUSTOMER_NEW_MERCHANTS_FEATURES)
        self.assertEqual(len(candidate.features), 55); self.assertEqual(candidate.model_params, parent.model_params)
        candidate_frame, fingerprint = load_training(candidate, root); parent_frame, _ = load_training(parent, root)
        pd.testing.assert_frame_equal(candidate_frame[parent.features], parent_frame[parent.features], check_dtype=True, check_exact=True)
        self.assertTrue(candidate_frame.transaction_id.astype(str).equals(parent_frame.transaction_id.astype(str)))
        _, _, splits = make_splits(candidate_frame, candidate, fingerprint, reuse=root / candidate.splits_file)
        self.assertEqual(splits["signature"], "6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47")


if __name__ == "__main__": unittest.main()
