"""Focused contract tests for the R037 prepared-parent model comparison."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.feature_experiment import FeatureSpec


class R037ContractTests(unittest.TestCase):
    def _payload(self):
        return json.loads((Path(__file__).parents[1] / "spec_r037.json").read_text(encoding="utf-8"))

    def test_exact_parent_and_no_feature_delta_are_required(self):
        spec = FeatureSpec.load(Path(__file__).parents[1] / "spec_r037.json")
        self.assertEqual(spec.parent, spec.prepared_parent)
        self.assertEqual(spec.added_features, ())
        for field, value in (("prepared_parent", "R017"), ("added_features", ["wrong_feature"]), ("feature_module", "src.velocity")):
            payload = self._payload(); payload[field] = value
            with TemporaryDirectory() as temporary:
                path = Path(temporary) / "spec.json"; path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError): FeatureSpec.load(path)

    def test_test_inference_and_non_model_override_are_rejected(self):
        for field, value in (("predict_test", True), ("model_override", {"model": "lightgbm", "model_params": {}, "features": ["x"]})):
            payload = self._payload(); payload[field] = value
            with TemporaryDirectory() as temporary:
                path = Path(temporary) / "spec.json"; path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError): FeatureSpec.load(path)


if __name__ == "__main__":
    unittest.main()
