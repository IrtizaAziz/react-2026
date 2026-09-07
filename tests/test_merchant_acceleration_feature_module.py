import unittest

from src import merchant_acceleration as module


class MerchantAccelerationFeatureModuleTests(unittest.TestCase):
    def test_certification(self):
        for check in module.certification_cases().values():
            check()


if __name__ == "__main__":
    unittest.main()
