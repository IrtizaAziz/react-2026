"""Tests for derived generic-experiment documentation only."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.experiment_docs import rebuild_experiment_index, render_experiment_card, write_experiment_card


class ExperimentDocumentationTests(unittest.TestCase):
    def _evidence(self):
        record = {"experiment_id": "R999", "timestamp": "2026-01-01T00:00:00+00:00", "status": "completed", "model": "catboost", "hypothesis": "h",
                  "parent_experiment": "R017", "provenance_path": "outputs/reports/R999/", "oof_path": "outputs/oof/R999.csv",
                  "config": {"features": ["base", "new"]}, "feature_experiment_provenance": {"normalized_spec": {"hypothesis": "h", "feature_module": "src.example", "added_features": ["new"]}, "split_signature": "split", "train_fingerprint": {"rows": 2}, "parent_config_sha256": "config", "module_source_sha256": {"module": "hash"}, "parent_matrix_cache": {"used": True, "matrix_sha256": "cache"}, "certification": {"results": {"oracle": True}}}}
        report = {"experiment_id": "R999", "parent": "R017", "incumbent": "R017", "metrics": {"F1": .7, "F2": .8, "F2_late_corrected": .6, "July_1_15": .5}, "delta_vs_parent": {"F2": .01}, "delta_vs_incumbent": {"F2": .01}, "classification": "WIN", "submission_gate_pass": False, "submission_gates": {"F2": True}, "early_stopped": False, "certification_reused": True, "feature_importances": {"F2": [{"feature": "new", "importance": 2.0}], "F1": []}, "runtime_seconds": 1.0, "test_inference": False, "submission_created": False}
        return record, report

    def test_card_is_deterministic_and_has_unknowns(self):
        record, report = self._evidence()
        with TemporaryDirectory() as temporary:
            first = render_experiment_card(temporary, record, report); second = render_experiment_card(temporary, record, report)
            self.assertEqual(first, second); self.assertIn("| F2 early | UNKNOWN", first); self.assertIn("feature importances are descriptive, not causal", first)
            path = write_experiment_card(temporary, record, report)
            self.assertTrue((Path(temporary) / path).exists())

    def test_index_uses_unknown_for_old_evidence_without_mutating_reports(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "experiments").mkdir(); (root / "outputs/reports/R001").mkdir(parents=True)
            fields = ["experiment_id", "parent_experiment", "model", "change_description", "public_lb"]
            with (root / "experiments/experiments.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerow({"experiment_id": "R001", "model": "catboost"})
            report = root / "outputs/reports/R001/decision_report.json"; report.write_text('{"metrics":{"F2":0.2}}', encoding="utf-8")
            before = report.read_bytes(); path = rebuild_experiment_index(root)
            self.assertEqual(before, report.read_bytes()); self.assertIn("UNKNOWN", path.read_text(encoding="utf-8"))


if __name__ == "__main__": unittest.main()
