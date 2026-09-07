import unittest

import pandas as pd

from src import device_recent_sharing as module


class DeviceRecentSharingFeatureModuleTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "timestamp": [0, 1, 2, 2, 86_401, 604_801],
            "customer_id": ["a", "a", "b", "c", "a", "d"],
            "device_id": ["x"] * 6,
        }, index=[11, 4, 19, 2, 8, 7])

    def test_manifest_subset_order_and_alignment(self):
        requested = ["device_other_customers_7d", "device_unique_customers_24h"]
        result = module.build_features(self.frame(), requested)
        self.assertEqual(list(result), requested)
        self.assertTrue(result.index.equals(self.frame().index))

    def test_validation_and_certification(self):
        with self.assertRaises(ValueError):
            module.build_features(self.frame(), [])
        with self.assertRaises(ValueError):
            module.build_features(self.frame(), ["not_available"])
        for case in module.certification_cases().values():
            case()

    def test_raw_frame_adapter_matches_historical_r015_engine(self):
        raw = self.frame()
        lifetime = module.device_global_production(raw.timestamp, raw.customer_id, raw.device_id)
        expected = module.device_recent_sharing_production(raw.timestamp, raw.customer_id, raw.device_id,
                                                           lifetime.device_prior_distinct_customer_count)
        actual = module.build_features(raw, list(module.AVAILABLE_FEATURES))
        pd.testing.assert_frame_equal(actual, expected.set_axis(raw.index), check_exact=True)


if __name__ == "__main__":
    unittest.main()
