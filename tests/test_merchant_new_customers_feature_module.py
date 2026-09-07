"""Compatibility contract for the R025 merchant new-customer feature module."""
import unittest

import pandas as pd

from src import merchant_new_customers as module


class MerchantNewCustomerFeatureModuleTests(unittest.TestCase):
    def _raw(self):
        return pd.DataFrame({"timestamp": [0, 0, 1, 1, 2], "customer_id": ["a", "a", "a", "b", "c"],
                             "merchant_id": ["m", "m", "m", "m", "m"]}, index=pd.Index([10, 11, 12, 13, 14], name="source_row"))

    def test_manifest_and_ordered_subset_are_exact_r025_family(self):
        expected = ("merchant_new_customers_24h", "merchant_new_customers_7d", "merchant_new_customer_share_24h",
                    "merchant_new_customer_share_7d", "merchant_seconds_since_last_new_customer")
        self.assertEqual(module.AVAILABLE_FEATURES, expected)
        raw = self._raw(); requested = [expected[3], expected[0]]
        built = module.build_features(raw, requested)
        direct = module.merchant_new_customers_production(raw.timestamp, raw.customer_id, raw.merchant_id).loc[:, requested].set_axis(raw.index)
        self.assertEqual(list(built.columns), requested); self.assertTrue(built.index.equals(raw.index)); self.assertEqual(len(built), len(raw))
        pd.testing.assert_frame_equal(built, direct)

    def test_rejects_invalid_requests(self):
        raw = self._raw()
        for requested in ([], [module.AVAILABLE_FEATURES[0], module.AVAILABLE_FEATURES[0]], ["not_a_feature"], "merchant_new_customers_24h"):
            with self.assertRaises(ValueError): module.build_features(raw, requested)

    def test_generic_certification_cases_succeed(self):
        cases = module.certification_cases()
        required = {"strict_past", "oracle", "equal_timestamp_isolation", "permutation_invariance", "future_independence", "label_independence", "chunk_vs_whole"}
        self.assertTrue(required <= set(cases))
        for check in cases.values(): check()


if __name__ == "__main__": unittest.main()
