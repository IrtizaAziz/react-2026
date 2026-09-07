"""Causal and immutable-parent checks for R015 device recent sharing."""
import inspect
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.data import REACT2026_DEVICE_RECENT_SHARING_FEATURES, load_training
from src.device_recent_sharing import (FEATURES, device_recent_sharing_chunked,
    device_recent_sharing_from_prior_stream, device_recent_sharing_production)
from src.validation import make_splits


class R015DeviceRecentSharingTests(unittest.TestCase):
    def _frame(self):
        # Seconds; rows 2 and 3 are a tied batch and must not see one another.
        return pd.DataFrame({"timestamp": [0, 1, 2, 2, 86401, 604801, 604802],
            "customer_id": ["a", "a", "b", "c", "a", "d", "e"],
            "device_id": ["x", "x", "x", "x", "x", "x", "x"],
            "device_prior_distinct_customer_count": [0, 1, 1, 1, 3, 3, 4],
            "fraud": [0, 1, 0, 1, 0, 1, 0]})

    def test_windows_ties_bounds_ratio_and_label_isolation(self):
        base = self._frame()
        out = device_recent_sharing_production(base.timestamp, base.customer_id,
            base.device_id, base.device_prior_distinct_customer_count)
        self.assertEqual(list(out), FEATURES)
        self.assertEqual(out.device_unique_customers_24h.tolist(), [0, 1, 1, 1, 2, 1, 1])
        self.assertEqual(out.device_unique_customers_7d.tolist(), [0, 1, 1, 1, 3, 3, 3])
        self.assertEqual(out.device_other_customers_24h.tolist(), [0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(out.device_other_customers_7d.tolist(), [0, 0, 1, 1, 2, 3, 3])
        np.testing.assert_allclose(out.device_7d_unique_to_lifetime_unique_ratio,
            [0, 1, 1, 1, 1, 1, .75])
        self.assertEqual(out.device_unique_customers_24h.iloc[2], out.device_unique_customers_24h.iloc[3])
        self.assertNotIn("fraud", inspect.signature(device_recent_sharing_production).parameters)
        self.assertNotIn("label", inspect.signature(device_recent_sharing_production).parameters)

    def test_future_permutation_chunk_and_continuation_parity(self):
        base = self._frame(); out = device_recent_sharing_production(base.timestamp, base.customer_id, base.device_id, base.device_prior_distinct_customer_count)
        future = pd.concat([base, pd.DataFrame({"timestamp": [700000], "customer_id": ["z"], "device_id": ["x"], "device_prior_distinct_customer_count": [5], "fraud": [1]})], ignore_index=True)
        pd.testing.assert_frame_equal(out, device_recent_sharing_production(future.timestamp, future.customer_id, future.device_id, future.device_prior_distinct_customer_count).iloc[:len(base)].reset_index(drop=True))
        pd.testing.assert_frame_equal(out, device_recent_sharing_chunked([base.iloc[:2], base.iloc[2:4], base.iloc[4:]]))
        continued = device_recent_sharing_from_prior_stream(base.iloc[:2].timestamp, base.iloc[:2].customer_id, base.iloc[:2].device_id, base.iloc[:2].device_prior_distinct_customer_count, base.iloc[2:].timestamp, base.iloc[2:].customer_id, base.iloc[2:].device_id, base.iloc[2:].device_prior_distinct_customer_count)
        pd.testing.assert_frame_equal(out.iloc[2:].reset_index(drop=True), continued)

    def test_r013_parent_manifest_and_saved_split(self):
        root = Path(ROOT); parent, candidate = load_config(root / "config_r013.json"), load_config(root / "config_r015.json")
        candidate.validate(); self.assertEqual(candidate.features[:len(parent.features)], parent.features)
        self.assertEqual(candidate.features[len(parent.features):], REACT2026_DEVICE_RECENT_SHARING_FEATURES)
        self.assertEqual(len(candidate.features), 50)
        frame, fingerprint = load_training(candidate, root); parent_frame, _ = load_training(parent, root)
        pd.testing.assert_frame_equal(frame[parent.features], parent_frame[parent.features], check_dtype=True, check_exact=True)
        _, _, splits = make_splits(frame, candidate, fingerprint, reuse=root / candidate.splits_file)
        self.assertEqual(splits["signature"], "6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47")


if __name__ == "__main__":
    unittest.main()
