"""Generic-interface parity tests for the unchanged R024 engine."""
import unittest
import pandas as pd

from src import device_location
from src.customer_relationships import device_global_production


class DeviceLocationFeatureModuleTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame({"timestamp": [0, 1, 2, 2, 3, 4], "customer_id": ["u", "u", "v", "w", "u", "u"],
                             "device_id": ["d", "d", "d", "d", "d", "d"], "location": [None, None, "a", "a", None, "a"],
                             "fraud": [0, 1, 0, 1, 0, 1]}, index=[10, 11, 12, 13, 14, 15])

    def test_requested_subset_matches_existing_production_path(self):
        frame = self._frame(); requested = ["device_location_share_of_device_history", "device_location_prior_count"]
        history = device_global_production(frame.timestamp, frame.customer_id, frame.device_id)
        expected = device_location.device_location_production(frame.timestamp, frame.device_id, frame.location, history.device_prior_count).loc[:, requested].set_axis(frame.index)
        actual = device_location.build_features(frame, requested)
        self.assertEqual(tuple(device_location.AVAILABLE_FEATURES), tuple(device_location.FEATURES))
        self.assertEqual(list(actual), requested); self.assertTrue(actual.index.equals(frame.index))
        pd.testing.assert_frame_equal(actual, expected)

    def test_subset_validation_and_certification(self):
        frame = self._frame()
        for requested in ([], ["missing"], [device_location.FEATURES[0], device_location.FEATURES[0]], "not-a-list"):
            with self.assertRaises(ValueError): device_location.build_features(frame, requested)
        for check in device_location.certification_cases().values(): check()


if __name__ == "__main__": unittest.main()
