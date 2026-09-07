"""Mandatory causal, parity, and locked-split preflight for R025."""
import inspect
from pathlib import Path
import unittest

import pandas as pd

from src.config import ROOT, load_config
from src.data import REACT2026_MERCHANT_NEW_CUSTOMERS_FEATURES, load_training
from src.merchant_new_customers import (FEATURES, merchant_new_customers_chunked,
    merchant_new_customers_production, merchant_new_customers_simple_oracle)
from src.validation import make_splits


class R025MerchantNewCustomerTests(unittest.TestCase):
    def _frame(self):
        day = 86_400
        return pd.DataFrame({"transaction_id": list("abcdefghijk"),
            "timestamp": [0, 0, day - 1, day - 1, day - 1, 2 * day - 1, 2 * day - 1, 8 * day, 8 * day, 9 * day, 9 * day],
            "customer_id": ["a", "a", "a", "b", "b", "a", "c", "a", "d", "e", "e"],
            "merchant_id": ["m", "m", "m", "m", "m", "m", "m", "m", "m", "m", "n"],
            "fraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]})

    def test_strict_past_ties_dedup_oracle_and_independence(self):
        day = 86_400; base = self._frame(); actual = merchant_new_customers_production(base.timestamp, base.customer_id, base.merchant_id)
        self.assertEqual(list(actual), FEATURES)
        # At t=day, both tied rows see only the one relationship created at t=0.
        self.assertEqual(actual.merchant_new_customers_24h.tolist()[:7], [0, 0, 1, 1, 1, 0, 0])
        self.assertEqual(actual.merchant_new_customers_7d.tolist()[:7], [0, 0, 1, 1, 1, 2, 2])
        self.assertAlmostEqual(actual.merchant_new_customer_share_24h.iloc[2], 1.0)
        self.assertTrue(actual.iloc[2].equals(actual.iloc[3]))  # equal-timestamp isolation
        self.assertNotIn("fraud", inspect.signature(merchant_new_customers_production).parameters)
        pd.testing.assert_frame_equal(actual, merchant_new_customers_simple_oracle(base.timestamp, base.customer_id, base.merchant_id))
        # Permuting tied rows cannot alter values associated with their transaction IDs.
        permuted = pd.concat([base.iloc[:2], base.iloc[[4, 2, 3]], base.iloc[5:]], ignore_index=True)
        got = merchant_new_customers_production(permuted.timestamp, permuted.customer_id, permuted.merchant_id)
        pd.testing.assert_frame_equal(pd.concat([base.transaction_id, actual], axis=1).set_index("transaction_id").sort_index(), pd.concat([permuted.transaction_id, got], axis=1).set_index("transaction_id").sort_index())
        future = pd.concat([base, pd.DataFrame({"transaction_id": ["z"], "timestamp": [99 * day], "customer_id": ["future_customer"], "merchant_id": ["future_merchant"], "fraud": [1]})], ignore_index=True)
        pd.testing.assert_frame_equal(actual, merchant_new_customers_production(future.timestamp, future.customer_id, future.merchant_id).iloc[:len(base)].reset_index(drop=True))
        # Future mutations of either relationship endpoint and labels cannot affect prior rows.
        mutated = future.copy(); mutated.loc[len(base), ["customer_id", "merchant_id"]] = ["a", "m"]
        pd.testing.assert_frame_equal(actual, merchant_new_customers_production(mutated.timestamp, mutated.customer_id, mutated.merchant_id).iloc[:len(base)].reset_index(drop=True))
        labels = future.copy(); labels["fraud"] = 1 - labels["fraud"]
        pd.testing.assert_frame_equal(merchant_new_customers_production(future.timestamp, future.customer_id, future.merchant_id), merchant_new_customers_production(labels.timestamp, labels.customer_id, labels.merchant_id))
        # The duplicate (b,m) rows at t=day create exactly one relationship event for later rows.
        self.assertEqual(actual.merchant_new_customers_7d.iloc[5], 2)
        pd.testing.assert_frame_equal(actual, merchant_new_customers_chunked([base.iloc[:3], base.iloc[3:7], base.iloc[7:]]))

    def test_r017_parity_row_ids_and_locked_split(self):
        root = Path(ROOT); parent, candidate = load_config(root / "config_r017.json"), load_config(root / "config_r025.json"); candidate.validate()
        self.assertEqual(candidate.features[:len(parent.features)], parent.features)
        self.assertEqual(candidate.features[len(parent.features):], REACT2026_MERCHANT_NEW_CUSTOMERS_FEATURES)
        self.assertEqual(len(candidate.features), 55); self.assertEqual(candidate.model_params, parent.model_params)
        candidate_frame, fingerprint = load_training(candidate, root); parent_frame, _ = load_training(parent, root)
        pd.testing.assert_frame_equal(candidate_frame[parent.features], parent_frame[parent.features], check_dtype=True, check_exact=True)
        self.assertTrue(candidate_frame.transaction_id.astype(str).equals(parent_frame.transaction_id.astype(str)))
        _, _, splits = make_splits(candidate_frame, candidate, fingerprint, reuse=root / candidate.splits_file)
        self.assertEqual(splits["signature"], "6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47")


if __name__ == "__main__": unittest.main()
