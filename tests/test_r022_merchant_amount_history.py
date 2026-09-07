"""Mandatory causal, parity, and locked-split preflight for R022."""
import inspect
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.config import ROOT, load_config
from src.data import load_training
from src.merchant_amount_history import FEATURES, merchant_amount_history_chunked, merchant_amount_history_production, merchant_amount_history_simple_oracle
from src.validation import make_splits


class R022MerchantAmountHistoryTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame({"transaction_id":list("abcdefgh"), "timestamp":[0,1,2,2,3,4,5,6], "merchant_id":["m","m","m","m","m","m","m","m"], "amount_bdt":[10.,20.,30.,40.,0.,50.,60.,70.], "fraud":[0,1,0,1,0,1,0,1]})

    def test_causal_semantics_oracle_and_safe_missing_conventions(self):
        base = self._frame(); actual = merchant_amount_history_production(base.timestamp, base.merchant_id, base.amount_bdt)
        self.assertEqual(list(actual), FEATURES); self.assertTrue(actual.iloc[0].isna().all()); self.assertAlmostEqual(actual.merchant_prior_mean_amount.iloc[1], 10.0)
        self.assertEqual(actual.merchant_prior_mean_amount.iloc[2], actual.merchant_prior_mean_amount.iloc[3])
        self.assertEqual(actual.merchant_prior_std_log_amount.iloc[1], 0.0); self.assertTrue(actual.merchant_log_amount_zscore.iloc[:5].isna().all())
        self.assertNotIn("fraud", inspect.signature(merchant_amount_history_production).parameters)
        pd.testing.assert_frame_equal(actual, merchant_amount_history_simple_oracle(base.timestamp, base.merchant_id, base.amount_bdt))
        permuted = pd.concat([base.iloc[:2], base.iloc[[3,2]], base.iloc[4:]], ignore_index=True)
        got = merchant_amount_history_production(permuted.timestamp, permuted.merchant_id, permuted.amount_bdt)
        pd.testing.assert_frame_equal(pd.concat([base.transaction_id, actual], axis=1).set_index("transaction_id").sort_index(), pd.concat([permuted.transaction_id, got], axis=1).set_index("transaction_id").sort_index())
        future = pd.concat([base, pd.DataFrame({"transaction_id":["z"],"timestamp":[99],"merchant_id":["m"],"amount_bdt":[999999.],"fraud":[0]})], ignore_index=True)
        pd.testing.assert_frame_equal(actual, merchant_amount_history_production(future.timestamp, future.merchant_id, future.amount_bdt).iloc[:len(base)].reset_index(drop=True))
        changed = future.copy(); changed.loc[len(base), "amount_bdt"] = 1.0
        pd.testing.assert_frame_equal(actual, merchant_amount_history_production(changed.timestamp, changed.merchant_id, changed.amount_bdt).iloc[:len(base)].reset_index(drop=True))
        labels = future.copy(); labels["fraud"] = 1 - labels["fraud"]
        pd.testing.assert_frame_equal(merchant_amount_history_production(future.timestamp, future.merchant_id, future.amount_bdt), merchant_amount_history_production(labels.timestamp, labels.merchant_id, labels.amount_bdt))
        pd.testing.assert_frame_equal(actual, merchant_amount_history_chunked([base.iloc[:2], base.iloc[2:4], base.iloc[4:]]))
        with self.assertRaises(ValueError): merchant_amount_history_production([0], ["m"], [np.nan])

    def test_r017_parity_row_ids_and_locked_split(self):
        root = Path(ROOT); parent, candidate = load_config(root / "config_r017.json"), load_config(root / "config_r022.json"); candidate.validate()
        self.assertEqual(candidate.features[:len(parent.features)], parent.features); self.assertEqual(candidate.features[len(parent.features):], FEATURES); self.assertEqual(len(candidate.features), 55); self.assertEqual(candidate.model_params, parent.model_params)
        candidate_frame, fingerprint = load_training(candidate, root); parent_frame, _ = load_training(parent, root)
        pd.testing.assert_frame_equal(candidate_frame[parent.features], parent_frame[parent.features], check_dtype=True, check_exact=True)
        self.assertTrue(candidate_frame.transaction_id.astype(str).equals(parent_frame.transaction_id.astype(str)))
        _, _, splits = make_splits(candidate_frame, candidate, fingerprint, reuse=root / candidate.splits_file)
        self.assertEqual(splits["signature"], "6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47")


if __name__ == "__main__": unittest.main()
