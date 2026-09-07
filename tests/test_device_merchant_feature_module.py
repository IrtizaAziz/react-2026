import unittest

import pandas as pd

from src import device_merchant as module


class DeviceMerchantFeatureModuleTests(unittest.TestCase):
    def test_subset_order_and_raw_alignment(self):
        frame = pd.DataFrame({"timestamp": [0, 1], "customer_id": ["a", "b"], "device_id": ["d", "d"], "merchant_id": ["m", "m"]}, index=[8, 3])
        result = module.build_features(frame, ["device_merchant_share_of_device_history"])
        self.assertEqual(list(result), ["device_merchant_share_of_device_history"])
        self.assertTrue(result.index.equals(frame.index))

    def test_validation_and_certification(self):
        frame = pd.DataFrame({"timestamp": [0], "customer_id": ["a"], "device_id": ["d"], "merchant_id": ["m"]})
        with self.assertRaises(ValueError): module.build_features(frame, [])
        with self.assertRaises(ValueError): module.build_features(frame, ["missing"])
        for case in module.certification_cases().values():
            case()


if __name__ == "__main__":
    unittest.main()
