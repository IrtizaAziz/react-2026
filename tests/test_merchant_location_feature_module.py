import unittest
import pandas as pd
from src import merchant_location as module


class MerchantLocationFeatureModuleTests(unittest.TestCase):
    def test_subset_order_and_raw_alignment(self):
        frame = pd.DataFrame({"timestamp": [0, 1], "customer_id": ["a", "b"], "merchant_id": ["m", "m"], "location": ["l", "l"]}, index=[8, 3])
        result = module.build_features(frame, ["merchant_location_share_of_merchant_history"])
        self.assertEqual(list(result), ["merchant_location_share_of_merchant_history"]); self.assertTrue(result.index.equals(frame.index))

    def test_certification(self):
        for check in module.certification_cases().values(): check()


if __name__ == "__main__": unittest.main()
