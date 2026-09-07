import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.react_runner import _next_id, record_failure


class FailureRecordingTests(unittest.TestCase):
    def _setup(self, root):
        (root / "specs").mkdir(); (root / "experiments").mkdir(); (root / "outputs/reports").mkdir(parents=True)
        fields = ["experiment_id", "timestamp", "parent_experiment", "model", "hypothesis", "change_description", "status"]
        with (root / "experiments/experiments.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
        (root / "outputs/reports/R029.json").write_text(json.dumps({"status": "completed", "model": "catboost"}), encoding="utf-8")
        spec = {"experiment_id": "R030", "parent": "R029", "feature_module": "src.example", "added_features": ["new"], "hypothesis": "h", "predict_test": False}
        path = root / "specs/r030.json"; path.write_text(json.dumps(spec), encoding="utf-8"); return path

    def test_failure_consumes_id_and_has_no_fake_metrics_or_report(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); spec = self._setup(root)
            record = record_failure(root, spec, "certification mismatch")
            self.assertEqual(record["status"], "failed")
            self.assertEqual(_next_id(root), "R031")
            self.assertFalse((root / "outputs/reports/R030/decision_report.json").exists())
            self.assertNotIn("metrics", record)

    def test_failure_cannot_be_recorded_twice_or_for_non_next_id(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); spec = self._setup(root)
            record_failure(root, spec, "first")
            with self.assertRaises(ValueError): record_failure(root, spec, "duplicate")
            bad = json.loads(spec.read_text(encoding="utf-8")); bad["experiment_id"] = "R032"; spec.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError): record_failure(root, spec, "wrong id")


if __name__ == "__main__": unittest.main()
