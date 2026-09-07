import unittest

import pandas as pd

from src import customer_temporal_habits as module


class CustomerTemporalHabitsTests(unittest.TestCase):
    def test_subset_order_and_raw_alignment(self):
        frame = pd.DataFrame({"timestamp": [0, 1, 2], "customer_id": ["a", "a", "b"]}, index=[8, 3, 5])
        result = module.build_features(frame, ["customer_weekday_share"])
        self.assertEqual(list(result), ["customer_weekday_share"])
        self.assertTrue(result.index.equals(frame.index))

    def test_validation_and_all_certification_cases(self):
        frame = pd.DataFrame({"timestamp": [0, 1], "customer_id": ["a", "a"]})
        with self.assertRaises(ValueError): module.build_features(frame, [])
        with self.assertRaises(ValueError): module.build_features(frame, ["missing"])
        for case in module.certification_cases().values():
            case()


if __name__ == "__main__":
    unittest.main()
