"""Mandatory causal, parity, and locked-split preflight for R024."""
import inspect
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.data import load_training
from src.device_location import FEATURES, device_location_chunked, device_location_production, device_location_simple_oracle
from src.validation import make_splits


class R024DeviceLocationTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame({"transaction_id": list("abcdefgh"), "timestamp": [0, 1, 2, 2, 3, 4, 5, 6],
                             "device_id": ["d", "d", "d", "d", "d", "x", "d", "d"],
                             "location": [None, None, "a", "a", None, "a", "a", "b"],
                             "device_prior_count": [0, 1, 2, 2, 4, 0, 5, 6], "fraud": [0, 1, 0, 1, 0, 1, 0, 1]})

    def test_causal_semantics_oracle_and_independence(self):
        base = self._frame()
        actual = device_location_production(base.timestamp, base.device_id, base.location, base.device_prior_count)
        self.assertEqual(list(actual), FEATURES)
        self.assertEqual(actual.device_location_prior_count.tolist(), [0, 1, 0, 0, 2, 0, 2, 0])
        self.assertEqual(actual.device_location_is_new.tolist(), [1, 0, 1, 1, 0, 1, 0, 1])
        self.assertAlmostEqual(actual.device_location_share_of_device_history.iloc[1], 1.0)
        self.assertAlmostEqual(actual.device_location_share_of_device_history.iloc[4], 0.5)
        self.assertTrue(np.isnan(actual.device_location_share_of_device_history.iloc[0]))
        # Equal-timestamp duplicate pairs query the same strictly-prior state.
        self.assertEqual(actual.device_location_prior_count.iloc[2], actual.device_location_prior_count.iloc[3])
        self.assertNotIn("fraud", inspect.signature(device_location_production).parameters)
        pd.testing.assert_frame_equal(actual, device_location_simple_oracle(base.timestamp, base.device_id, base.location, base.device_prior_count))
        permuted = pd.concat([base.iloc[:2], base.iloc[[3, 2]], base.iloc[4:]], ignore_index=True)
        got = device_location_production(permuted.timestamp, permuted.device_id, permuted.location, permuted.device_prior_count)
        pd.testing.assert_frame_equal(pd.concat([base.transaction_id, actual], axis=1).set_index("transaction_id").sort_index(), pd.concat([permuted.transaction_id, got], axis=1).set_index("transaction_id").sort_index())
        future = pd.concat([base, pd.DataFrame({"transaction_id": ["z"], "timestamp": [99], "device_id": ["future-device"], "location": ["future-location"], "device_prior_count": [0], "fraud": [0]})], ignore_index=True)
        expected = actual
        pd.testing.assert_frame_equal(expected, device_location_production(future.timestamp, future.device_id, future.location, future.device_prior_count).iloc[:len(base)].reset_index(drop=True))
        mutated = future.copy(); mutated.loc[len(base), ["device_id", "location"]] = ["changed-device", "changed-location"]
        pd.testing.assert_frame_equal(expected, device_location_production(mutated.timestamp, mutated.device_id, mutated.location, mutated.device_prior_count).iloc[:len(base)].reset_index(drop=True))
        labels = future.copy(); labels["fraud"] = 1 - labels["fraud"]
        pd.testing.assert_frame_equal(device_location_production(future.timestamp, future.device_id, future.location, future.device_prior_count), device_location_production(labels.timestamp, labels.device_id, labels.location, labels.device_prior_count))
        pd.testing.assert_frame_equal(expected, device_location_chunked([base.iloc[:2], base.iloc[2:4], base.iloc[4:]]))

    def test_r017_parity_alignment_split_and_missing_sentinel(self):
        root = Path(ROOT); parent, candidate = load_config(root / "config_r017.json"), load_config(root / "config_r024.json"); candidate.validate()
        self.assertEqual(candidate.features[:len(parent.features)], parent.features); self.assertEqual(candidate.features[len(parent.features):], FEATURES)
        self.assertEqual(len(parent.features), 50); self.assertEqual(len(candidate.features), 54); self.assertEqual(candidate.model, parent.model); self.assertEqual(candidate.model_params, parent.model_params)
        candidate_frame, fingerprint = load_training(candidate, root); parent_frame, _ = load_training(parent, root)
        pd.testing.assert_frame_equal(candidate_frame[parent.features], parent_frame[parent.features], check_dtype=True, check_exact=True)
        self.assertTrue(candidate_frame.transaction_id.astype(str).equals(parent_frame.transaction_id.astype(str)))
        raw = pd.read_csv(root / "train.csv", dtype={"transaction_id": "string"})
        sentinel = raw.location.isna().to_numpy()
        self.assertTrue(np.array_equal(candidate_frame.loc[sentinel, "customer_location_prior_count"].to_numpy(), candidate_frame.loc[sentinel, "customer_location_prior_count"].to_numpy()))
        # Direct missing-location parity: both histories treat null as one durable category.
        tiny = pd.DataFrame({"timestamp": [0, 1], "device_id": ["d", "d"], "location": [None, None], "device_prior_count": [0, 1]})
        self.assertEqual(device_location_production(tiny.timestamp, tiny.device_id, tiny.location, tiny.device_prior_count).device_location_prior_count.tolist(), [0, 1])
        _, _, splits = make_splits(candidate_frame, candidate, fingerprint, reuse=root / candidate.splits_file)
        self.assertEqual(splits["signature"], "6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47")


if __name__ == "__main__":
    unittest.main()
