import unittest

import pandas as pd

from src import customer_account_consistency as module


class CustomerAccountConsistencyFeatureModuleTests(unittest.TestCase):
    def test_subset_order_and_raw_alignment(self):
        frame = pd.DataFrame({"timestamp": [0, 1], "customer_id": ["a", "a"], "account_age_days": [0, 1]}, index=[8, 3])
        result = module.build_features(frame, ["customer_creation_day_mode_share"])
        self.assertEqual(list(result), ["customer_creation_day_mode_share"])
        self.assertTrue(result.index.equals(frame.index))

    def test_all_certification_cases(self):
        for case in module.certification_cases().values():
            case()


if __name__ == "__main__":
    unittest.main()
