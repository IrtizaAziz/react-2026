"""Isolated synthetic plumbing checks; no external data or competition experiments."""
import contextlib
import copy
import csv
from dataclasses import replace
import importlib
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.audit import audit, sampled_table
from src.config import Config, ROOT, load_config
from src.compare import comparison
from src.customer_history import FEATURES as CUSTOMER_HISTORY_FEATURES, customer_history_chunked, customer_history_from_prior_stream, customer_history_oracle, customer_history_production
from src.customer_relationships import FEATURES as CUSTOMER_RELATIONSHIP_FEATURES, customer_relationship_chunked, customer_relationship_from_prior_stream, customer_relationship_oracle, customer_relationship_production
from src.customer_relationships import DEVICE_GLOBAL_FEATURES, device_global_chunked, device_global_from_prior_stream, device_global_oracle, device_global_production
from src.velocity import FEATURES as VELOCITY_FEATURES, velocity_chunked, velocity_from_prior_stream, velocity_oracle, velocity_production
from src.velocity import BROAD_FEATURES, broad_velocity_chunked, broad_velocity_oracle, broad_velocity_production
from src.ensemble import blend_experiments, compatible_artifacts
from src.export_notebook import export_notebook, code, markdown, notebook
from src.features import build_pipeline
from src.metrics import score, metric_definition
from src.predict import predict_experiment, read_predictions, save_predictions
from src.replay import compare_predictions, replay_oof_score
from src.submission import build_submission, generate_submission, validate_probabilities
from src.train import _catboost_model_feature_order, _feature_importance, _lightgbm_transformed_feature_order, train_experiment
from src.utils import checked_record, read_json, save_json, sha256
from src.validation import make_splits
from src.temporal import past_group_count, past_group_mean
from src.data import REACT2026_STATIC_FEATURES, REACT2026_CUSTOMER_HISTORY_FEATURES, REACT2026_CUSTOMER_RELATIONSHIP_FEATURES, REACT2026_DEVICE_GLOBAL_FEATURES, REACT2026_VELOCITY_FEATURES, REACT2026_MERCHANT_ONEHOT_PROFILE


class StarterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="react-synthetic-")
        cls.root = Path(cls.temporary.name)
        raw = cls.root / "data/raw"
        raw.mkdir(parents=True)
        cls.training = pd.DataFrame({"id": [f"{i:03d}" for i in range(12)],
                                     "number": np.arange(12, dtype=float),
                                     "category": [f"unique_{i}" for i in range(12)],
                                     "target": ["no", "yes"] * 6})
        cls.training.loc[2, "number"] = np.nan
        cls.test = pd.DataFrame({"id": ["101", "102", "103"], "number": [2., 5., np.nan], "category": ["new", "unique_1", None]})
        cls.sample = pd.DataFrame({"id": ["103", "101", "102"], "target": [0., 0., 0.]})
        cls.training.to_csv(raw / "train.csv", index=False)
        cls.test.to_csv(raw / "test.csv", index=False)
        cls.sample.to_csv(raw / "sample_submission.csv", index=False)
        cls.config = Config(train_file="data/raw/train.csv", test_file="data/raw/test.csv", sample_file="data/raw/sample_submission.csv",
                            target="target", id_columns=["id"], features=["number", "category"], task="classification",
                            metric="log_loss", metric_direction="lower", prediction_kind="probability", class_order=["no", "yes"],
                            validation_type="stratified", validation_rationale="Synthetic plumbing test only, no competition strategy", n_splits=2,
                            shuffle=True, seed=42, model="logistic", submission_columns=["target"], submission_kind="probability",
                            submission_alignment="id", positive_class="yes", smoke_test=True)
        with contextlib.redirect_stdout(io.StringIO()), patch("src.predict.load_test", side_effect=AssertionError("Test data was opened during training")):
            cls.record = train_experiment(cls.config, "SMOKE001", "Check plumbing", "Tiny synthetic integration fixture", root=cls.root)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.record = predict_experiment(cls.root, "SMOKE001", aggregation="mean")
        cls.submission = generate_submission(cls.root, "SMOKE001", description="smoke")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_imports_and_help(self):
        for name in ("config", "audit", "data", "validation", "metrics", "features", "train", "predict", "ensemble", "submission", "utils", "export_notebook"):
            importlib.import_module("src." + name)
        for name in ("audit", "train", "predict", "ensemble", "submission", "export_notebook"):
            result = subprocess.run([sys.executable, str(ROOT / "src" / f"{name}.py"), "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)

    def test_unset_configuration_fails_before_reserving_id(self):
        with self.assertRaisesRegex(ValueError, "Configure after launch"):
            train_experiment(Config(), "E001", "question", "change", root=self.root)
        self.assertFalse((self.root / "outputs/reports/E001.json").exists())

    def test_core_result_and_fold_local_preprocessing(self):
        record = checked_record(self.root, "SMOKE001")
        self.assertNotIn(b"\r\n", self.submission.read_bytes())
        self.assertNotIn(b"\r\n", (self.root / record["oof_path"]).read_bytes())
        self.assertEqual(record["oof_coverage"], 1)
        self.assertEqual(len(record["fold_scores"]), 2)
        saved = read_json(self.root / record["splits_path"])
        for fold, pair in enumerate(saved["splits"]):
            with (self.root / record["model_paths"][fold]).open("rb") as handle:
                model = pickle.load(handle)
            learned = set(model.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encode"].categories_[0])
            validation_only = set(self.training.iloc[pair["valid"]]["category"])
            self.assertFalse(learned & validation_only)
        frame, values, meta = read_predictions(self.root / record["prediction_path"])
        self.assertEqual(frame["id"].tolist(), ["101", "102", "103"])
        np.testing.assert_allclose(values.sum(axis=1), 1)
        self.assertEqual(meta["class_order"], ["no", "yes"])
        output = pd.read_csv(self.submission, dtype={"id": "string"})
        self.assertEqual(output["id"].tolist(), ["103", "101", "102"])
        np.testing.assert_allclose(output["target"], values[[2, 0, 1], 1])
        with (self.root / "experiments/experiments.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        row = next(r for r in rows if r["experiment_id"] == "SMOKE001")
        self.assertEqual(row["public_lb"], "")
        self.assertEqual(row["conclusion"], "")

    def test_no_overwrites(self):
        before = sha256(self.submission)
        with self.assertRaises(FileExistsError):
            generate_submission(self.root, "SMOKE001", description="smoke")
        self.assertEqual(before, sha256(self.submission))
        with self.assertRaises(FileExistsError):
            train_experiment(self.config, "SMOKE001", "same ID", "must stop", root=self.root)
        with self.assertRaises(FileExistsError):
            predict_experiment(self.root, "SMOKE001", aggregation="mean")

    def test_failed_run_keeps_id(self):
        with patch("src.train.clone", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                train_experiment(self.config, "SMOKE099", "Failure handling", "Injected failure before fit", root=self.root)
        self.assertEqual(read_json(self.root / "outputs/reports/SMOKE099.json")["status"], "failed")
        with self.assertRaises(FileExistsError):
            train_experiment(self.config, "SMOKE099", "Retry", "ID must remain reserved", root=self.root)

    def test_metrics_and_probability_contracts(self):
        regression = Config(metric="rmse", metric_direction="lower", prediction_kind="value")
        self.assertAlmostEqual(score([1, 3], [2, 1], regression), np.sqrt(2.5))
        self.assertAlmostEqual(score(["no", "yes"], [[.8, .2], [.1, .9]], self.config), -np.log(.8*.9)/2)
        accuracy = replace(self.config, metric="accuracy", metric_direction="higher", label_threshold=.5)
        self.assertEqual(score(["no", "yes"], [[.8, .2], [.1, .9]], accuracy), 1)
        auc = replace(self.config, metric="roc_auc", metric_direction="higher")
        self.assertEqual(score(["no", "yes"], [[.8, .2], [.1, .9]], auc), 1)
        for values in ([[1.1, -.1]], [[np.nan, 1]], [[.1, .1]]):
            with self.assertRaises(ValueError):
                validate_probabilities(values, sums=True)
        with self.assertRaises(ValueError):
            metric_definition(replace(self.config, metric="f1", metric_direction="higher"))
        with self.assertRaises(ValueError):
            metric_definition(replace(self.config, metric_direction="higher"))

    def test_replay_accepts_equivalent_predictions_with_different_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            frame = pd.DataFrame({"id": ["a", "b"]})
            values = np.array([[.9, .1], [.2, .8]])
            expected, actual = directory / "expected.csv", directory / "actual.csv"
            save_predictions(expected, frame, values, self.config, {"synthetic": True})
            save_predictions(actual, frame, values, self.config, {"synthetic": True})
            actual.write_bytes(actual.read_bytes().replace(b"\n", b"\r\n"))
            metadata = read_json(actual.with_suffix(".meta.json"))
            metadata["sha256"] = sha256(actual)
            save_json(actual.with_suffix(".meta.json"), metadata)
            report = compare_predictions(expected, actual, rtol=1e-7, atol=1e-9)
            self.assertNotEqual(report["expected_sha256"], report["actual_sha256"])
            self.assertTrue(report["reproduced"])
            self.assertEqual(report["values_outside_tolerance"], 0)

    def test_replay_rejects_predictions_outside_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            frame = pd.DataFrame({"id": ["a", "b"]})
            expected, actual = directory / "expected.csv", directory / "actual.csv"
            save_predictions(expected, frame, np.array([[.9, .1], [.2, .8]]), self.config, {"synthetic": True})
            save_predictions(actual, frame, np.array([[.9, .1], [.2, .8001]]), self.config, {"synthetic": True})
            report = compare_predictions(expected, actual, rtol=1e-7, atol=1e-9)
            self.assertFalse(report["reproduced"])
            self.assertGreater(report["values_outside_tolerance"], 0)

    def test_submission_rejects_ambiguity_and_bad_ids(self):
        frame, predictions, metadata = read_predictions(self.root / self.record["prediction_path"])
        duplicate = self.sample.copy()
        duplicate.loc[1, "id"] = duplicate.loc[0, "id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            build_submission(duplicate, frame, predictions, metadata, self.config)
        missing = self.sample.copy()
        missing.loc[0, "id"] = "999"
        with self.assertRaisesRegex(ValueError, "different IDs"):
            build_submission(missing, frame, predictions, metadata, self.config)
        with self.assertRaises(ValueError):
            build_submission(self.sample, frame, predictions, metadata, replace(self.config, submission_alignment=None))
        with self.assertRaises(ValueError):
            build_submission(self.sample, frame, predictions, metadata, replace(self.config, submission_columns=["wrong"]))

    def test_fraud_submission_contract(self):
        config = replace(self.config, target="fraud", id_columns=["transaction_id"], submission_columns=["fraud"],
                         submission_kind="probability", submission_alignment="id", positive_class="yes")
        sample = pd.DataFrame({"transaction_id": ["b", "a"], "fraud": [0., 0.]})
        artifact = pd.DataFrame({"__row__": [0, 1], "transaction_id": ["a", "b"]})
        predictions = np.array([[.1, .9], [.8, .2]])
        metadata = {"prediction_kind": "probability", "class_order": ["no", "yes"], "id_columns": ["transaction_id"]}
        output = build_submission(sample, artifact, predictions, metadata, config)
        self.assertEqual(list(output.columns), ["transaction_id", "fraud"])
        np.testing.assert_allclose(output["fraud"], [.2, .9])
        with self.assertRaises(ValueError):
            build_submission(sample, artifact, np.array([[-1, 2], [.8, .2]]), metadata, config)

    def test_splits_reuse_groups_and_time(self):
        record = checked_record(self.root, "SMOKE001")
        split_path = self.root / record["splits_path"]
        first = make_splits(self.training, self.config, record["train_fingerprint"], reuse=split_path)
        second = make_splits(self.training, self.config, record["train_fingerprint"])
        self.assertEqual(first[2], second[2])
        with self.assertRaisesRegex(ValueError, "do not match"):
            make_splits(self.training, self.config, {"changed": True}, reuse=split_path)
        frame = self.training.assign(group=np.repeat(np.arange(6), 2), time=np.arange(12)[::-1])
        for kind in ("kfold", "group", "stratified_group"):
            config = replace(self.config, validation_type=kind, group_column="group", shuffle=False)
            pairs, _, _ = make_splits(frame, config, {"synthetic": True})
            for training, valid in pairs:
                self.assertFalse(set(training) & set(valid))
                if "group" in kind:
                    self.assertFalse(set(frame.iloc[training].group) & set(frame.iloc[valid].group))
        temporal = replace(self.config, validation_type="time", shuffle=False, time_column="time", time_gap=1,
                           task="regression", prediction_kind="value", class_order=[])
        pairs, assignments, _ = make_splits(frame, temporal, {"synthetic": True})
        self.assertTrue((assignments == -1).any())
        for training, valid in pairs:
            self.assertLess(frame.iloc[training].time.max(), frame.iloc[valid].time.min())
        frame["time"] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ValueError, "tied"):
                make_splits(frame, temporal, {"synthetic": True})
        with patch("src.validation.model_selection.StratifiedGroupKFold", None):
            with self.assertRaisesRegex(ImportError, "no fallback"):
                make_splits(frame, replace(self.config, validation_type="stratified_group", group_column="group"), {})

    def test_time_holdout_and_past_only_helpers(self):
        frame = self.training.assign(time=np.arange(12), entity=["a", "a", "b", "b", "a", "a", "b", "b", "a", "a", "b", "b"], value=np.arange(12, dtype=float))
        temporal = replace(self.config, validation_type="time_holdout", validation_rationale="Synthetic chronological holdout", n_splits=1,
                           shuffle=False, time_column="time", time_valid_fraction=.25, time_gap=1)
        pairs, assignment, _ = make_splits(frame, temporal, {"synthetic": True})
        train, valid = pairs[0]
        self.assertLess(frame.iloc[train].time.max(), frame.iloc[valid].time.min())
        self.assertTrue((assignment[valid] == 0).all())
        self.assertEqual(past_group_count(frame, "entity", "time").tolist()[:4], [0, 1, 0, 1])
        means = past_group_mean(frame, "entity", "value", "time")
        self.assertTrue(np.isnan(means.iloc[0]))
        self.assertEqual(means.iloc[1], 0)

    def test_calendar_time_split_integrity(self):
        frame = pd.DataFrame({"id": ["a", "b", "c", "d", "e", "f"],
                              "target": [0, 1, 0, 1, 0, 1],
                              "timestamp": ["2026-01-01", "2026-03-13 23:59:59", "2026-03-14",
                                            "2026-05-14 23:59:59", "2026-05-15", "2026-07-15 23:59:59"]})
        config = replace(self.config, validation_type="calendar_time", shuffle=False, time_column="timestamp", n_splits=2,
                         class_order=[0, 1], positive_class=1,
                         calendar_folds=[{"train_before": "2026-03-14", "valid_start": "2026-03-14", "valid_end": "2026-05-15"},
                                         {"train_before": "2026-05-15", "valid_start": "2026-05-15", "valid_end": "2026-07-16"}])
        config.validate()
        pairs, assignment, payload = make_splits(frame, config, {"synthetic": "calendar"})
        self.assertEqual([(len(a), len(b)) for a, b in pairs], [(2, 2), (4, 2)])
        self.assertEqual(assignment.tolist(), [-1, -1, 0, 0, 1, 1])
        self.assertEqual(payload["splits"][1]["valid_time_min"], "2026-05-15 00:00:00+00:00")

    def test_average_precision_positive_class_and_partial_oof_replay(self):
        config = replace(self.config, train_file="train.csv", target="fraud", id_columns=["transaction_id"],
                         features=["number"], task="classification", metric="average_precision", metric_direction="higher",
                         prediction_kind="probability", class_order=[0, 1], positive_class=1,
                         validation_type="calendar_time", validation_rationale="synthetic", n_splits=2, shuffle=False,
                         time_column="timestamp", calendar_folds=[{"train_before": "2026-01-02", "valid_start": "2026-01-02", "valid_end": "2026-01-03"},
                                                                      {"train_before": "2026-01-03", "valid_start": "2026-01-03", "valid_end": "2026-01-04"}], model="logistic")
        self.assertEqual(score([0, 1], [[.9, .1], [.1, .9]], config), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = pd.DataFrame({"transaction_id": list("abcde"), "timestamp": pd.date_range("2026-01-01", periods=5, freq="D"),
                                     "number": np.arange(5), "fraud": [0, 0, 1, 0, 1]})
            training.to_csv(root / "train.csv", index=False)
            predictions = np.array([[np.nan, np.nan], [np.nan, np.nan], [.9, .1], [.8, .2], [.1, .9]])
            oof_path = root / "oof.csv"
            save_predictions(oof_path, training, predictions, config, {"synthetic": True}, folds=np.array([-1, -1, 0, 0, 1]), split_signature="split")
            record = {"oof_path": "oof.csv", "fold_scores": [0.5, 1.0], "cv_mean": 0.75, "split_signature": "split"}
            replay = replay_oof_score(record, record, root, config, rtol=1e-9, atol=1e-12)
            self.assertTrue(replay["within_tolerance"])
            self.assertEqual(replay["replayed_fold_scores"], [0.5, 1.0])
            self.assertIn("pooled_covered_oof_score", replay)

    def test_static_feature_importance_uses_explicit_feature_order(self):
        class Model:
            def get_feature_importance(self):
                return np.array([.2, .8])
        class Preprocess:
            def get_feature_names_out(self):
                raise AssertionError("Static importance must not use sklearn feature-name introspection")
        class Pipeline:
            named_steps = {"model": Model(), "preprocess": Preprocess()}
        result = _feature_importance(Pipeline(), ["amount_bdt", "hour"])
        self.assertEqual(result, [{"feature": "hour", "importance": .8}, {"feature": "amount_bdt", "importance": .2}])
        catboost_config = replace(self.config, model="catboost", features=["amount_bdt", "merchant_category", "hour"],
                                  categorical_features=["merchant_category"])
        self.assertEqual(_catboost_model_feature_order(catboost_config), ["amount_bdt", "hour", "merchant_category"])

    def test_customer_history_is_strictly_past_and_matches_oracle(self):
        base = pd.DataFrame({"transaction_id": ["a", "b", "c", "d", "e", "f", "g"],
                             "timestamp": [1, 1, 2, 3600, 3600, 7200, 7201],
                             "customer_id": ["x", "x", "x", "x", "y", "x", "y"],
                             "amount_bdt": [10., 30., 50., 70., 9., 90., 12.],
                             "fraud": [0, 1, 0, 1, 0, 1, 0]})
        oracle = customer_history_oracle(base.timestamp, base.customer_id, base.amount_bdt)
        production = customer_history_production(base.timestamp, base.customer_id, base.amount_bdt)
        pd.testing.assert_frame_equal(oracle, production)
        self.assertEqual(production.customer_prior_count.tolist()[:3], [0, 0, 2])
        self.assertTrue(np.isnan(production.customer_prior_mean_amount.iloc[0]))
        self.assertTrue(np.isnan(production.customer_prior_mean_amount.iloc[1]))
        self.assertEqual(production.customer_prior_mean_amount.iloc[2], 20.)
        self.assertEqual(production.customer_seconds_since_last.iloc[3], 3598.)
        shuffled = pd.concat([base.iloc[[1, 0]], base.iloc[2:]], ignore_index=True)
        shuffled_features = customer_history_production(shuffled.timestamp, shuffled.customer_id, shuffled.amount_bdt)
        original_by_id = pd.concat([base[["transaction_id"]], production], axis=1).set_index("transaction_id").sort_index()
        shuffled_by_id = pd.concat([shuffled[["transaction_id"]], shuffled_features], axis=1).set_index("transaction_id").sort_index()
        pd.testing.assert_frame_equal(original_by_id, shuffled_by_id)
        changed = base.copy(); changed.loc[1, "amount_bdt"] = 999.
        changed_features = customer_history_production(changed.timestamp, changed.customer_id, changed.amount_bdt)
        pd.testing.assert_series_equal(production.iloc[0], changed_features.iloc[0])
        future = pd.concat([base, pd.DataFrame({"transaction_id": ["h"], "timestamp": [9999], "customer_id": ["x"], "amount_bdt": [1.], "fraud": [1]})], ignore_index=True)
        future_features = customer_history_production(future.timestamp, future.customer_id, future.amount_bdt)
        pd.testing.assert_frame_equal(production, future_features.iloc[:len(base)].reset_index(drop=True))
        permuted_fraud = base.assign(fraud=1 - base.fraud)
        pd.testing.assert_frame_equal(production, customer_history_production(permuted_fraud.timestamp, permuted_fraud.customer_id, permuted_fraud.amount_bdt))
        chunked = customer_history_chunked([base.iloc[:2], base.iloc[2:5], base.iloc[5:]])
        pd.testing.assert_frame_equal(production, chunked)
        continued = customer_history_from_prior_stream(
            base.iloc[:3].timestamp, base.iloc[:3].customer_id, base.iloc[:3].amount_bdt,
            base.iloc[3:].timestamp, base.iloc[3:].customer_id, base.iloc[3:].amount_bdt,
        )
        pd.testing.assert_frame_equal(production.iloc[3:].reset_index(drop=True), continued)
        self.assertEqual(set(production.columns), set(CUSTOMER_HISTORY_FEATURES))
        for timestamps, customers, amounts in [
            ([0, 3600, 3600, 7200], ["a", "b", "a", "a"], [1., 2., 3., 4.]),
            ([5, 5, 6, 8, 8, 9], ["a", "b", "a", "b", "a", "b"], [5., 4., 3., 2., 1., 0.]),
        ]:
            pd.testing.assert_frame_equal(
                customer_history_oracle(timestamps, customers, amounts),
                customer_history_production(timestamps, customers, amounts),
            )

    def test_customer_relationships_are_strictly_past_and_match_oracle(self):
        base = pd.DataFrame({"transaction_id": ["a", "b", "c", "d", "e", "f", "g"],
                             "timestamp": [1, 1, 2, 2, 3600, 3600, 7200],
                             "customer_id": ["x", "x", "x", "x", "x", "y", "x"],
                             "device_id": ["d1", "d1", "d1", "d2", "d1", "d1", "d1"],
                             "location": ["l1", "l1", "l1", "l2", "l1", "l1", "l1"],
                             "fraud": [0, 1, 0, 1, 0, 1, 0]})
        oracle = customer_relationship_oracle(base.timestamp, base.customer_id, base.device_id, base.location)
        production = customer_relationship_production(base.timestamp, base.customer_id, base.device_id, base.location)
        pd.testing.assert_frame_equal(oracle, production)
        self.assertEqual(production.customer_device_prior_count.tolist()[:4], [0, 0, 2, 0])
        self.assertEqual(production.customer_location_prior_count.tolist()[:4], [0, 0, 2, 0])
        self.assertTrue(np.isnan(production.customer_device_count_share.iloc[0]))
        self.assertEqual(production.customer_device_count_share.iloc[2], 1.)
        shuffled = pd.concat([base.iloc[[1, 0]], base.iloc[2:]], ignore_index=True)
        shuffled_features = customer_relationship_production(shuffled.timestamp, shuffled.customer_id, shuffled.device_id, shuffled.location)
        original_by_id = pd.concat([base[["transaction_id"]], production], axis=1).set_index("transaction_id").sort_index()
        shuffled_by_id = pd.concat([shuffled[["transaction_id"]], shuffled_features], axis=1).set_index("transaction_id").sort_index()
        pd.testing.assert_frame_equal(original_by_id, shuffled_by_id)
        changed = base.copy(); changed.loc[1, "device_id"] = "d9"; changed.loc[1, "location"] = "l9"
        pd.testing.assert_series_equal(production.iloc[0], customer_relationship_production(changed.timestamp, changed.customer_id, changed.device_id, changed.location).iloc[0])
        future = pd.concat([base, base.iloc[[0]].assign(timestamp=9999, transaction_id="h")], ignore_index=True)
        pd.testing.assert_frame_equal(production, customer_relationship_production(future.timestamp, future.customer_id, future.device_id, future.location).iloc[:len(base)].reset_index(drop=True))
        permuted = base.assign(fraud=1 - base.fraud)
        pd.testing.assert_frame_equal(production, customer_relationship_production(permuted.timestamp, permuted.customer_id, permuted.device_id, permuted.location))
        chunked = customer_relationship_chunked([base.iloc[:2], base.iloc[2:5], base.iloc[5:]])
        pd.testing.assert_frame_equal(production, chunked)
        continued = customer_relationship_from_prior_stream(base.iloc[:2].timestamp, base.iloc[:2].customer_id, base.iloc[:2].device_id, base.iloc[:2].location,
                                                            base.iloc[2:].timestamp, base.iloc[2:].customer_id, base.iloc[2:].device_id, base.iloc[2:].location)
        pd.testing.assert_frame_equal(production.iloc[2:].reset_index(drop=True), continued)
        self.assertEqual(set(production.columns), set(CUSTOMER_RELATIONSHIP_FEATURES))

    def test_device_global_history_is_strictly_past_and_matches_oracle(self):
        # Includes globally unseen devices, a known device/new customer, tied first
        # uses by two customers, and repeated same-device rows in a tied batch.
        base = pd.DataFrame({"transaction_id": list("abcdefgh"),
                             "timestamp": [1, 2, 3, 3, 4, 4, 5, 6],
                             "customer_id": ["a", "a", "b", "c", "a", "a", "b", "d"],
                             "device_id": ["new", "shared", "shared", "shared", "shared", "shared", "shared", "new"],
                             "fraud": [0, 1, 0, 1, 0, 1, 0, 1]})
        oracle = device_global_oracle(base.timestamp, base.customer_id, base.device_id)
        production = device_global_production(base.timestamp, base.customer_id, base.device_id)
        pd.testing.assert_frame_equal(oracle, production)
        self.assertEqual(production.device_prior_count.tolist()[:4], [0, 0, 1, 1])
        self.assertEqual(production.device_prior_distinct_customer_count.tolist()[2:4], [1, 1])
        self.assertEqual(production.device_prior_distinct_customer_count_excluding_current.tolist()[2:4], [1, 1])
        self.assertTrue(np.isnan(production.device_seconds_since_last.iloc[0]))
        self.assertTrue(np.isnan(production.customer_share_of_device_prior_transactions.iloc[0]))
        # The tied rows at t=4 observe the identical pre-t=4 state.
        self.assertEqual(production.device_prior_count.iloc[4], production.device_prior_count.iloc[5])
        self.assertEqual(production.device_prior_distinct_customer_count.iloc[4], production.device_prior_distinct_customer_count.iloc[5])
        self.assertEqual(production.device_prior_distinct_customer_count.iloc[6], 3)
        self.assertEqual(production.device_prior_other_customer_transaction_count.iloc[4], 2)
        self.assertAlmostEqual(production.customer_share_of_device_prior_transactions.iloc[4], 1 / 3)
        # Permuting an equal-timestamp batch, future rows, and labels cannot alter
        # preceding features; output is aligned by the stable transaction identifier.
        shuffled = pd.concat([base.iloc[:2], base.iloc[[3, 2]], base.iloc[4:]], ignore_index=True)
        shuffled_features = device_global_production(shuffled.timestamp, shuffled.customer_id, shuffled.device_id)
        pd.testing.assert_frame_equal(pd.concat([base[["transaction_id"]], production], axis=1).set_index("transaction_id").sort_index(),
                                      pd.concat([shuffled[["transaction_id"]], shuffled_features], axis=1).set_index("transaction_id").sort_index())
        future = pd.concat([base, base.iloc[[0]].assign(timestamp=999, transaction_id="z")], ignore_index=True)
        pd.testing.assert_frame_equal(production, device_global_production(future.timestamp, future.customer_id, future.device_id).iloc[:len(base)].reset_index(drop=True))
        pd.testing.assert_frame_equal(production, device_global_production(base.timestamp, base.customer_id, base.device_id))
        chunked = device_global_chunked([base.iloc[:3], base.iloc[3:6], base.iloc[6:]])
        pd.testing.assert_frame_equal(production, chunked)
        continued = device_global_from_prior_stream(base.iloc[:4].timestamp, base.iloc[:4].customer_id, base.iloc[:4].device_id,
                                                     base.iloc[4:].timestamp, base.iloc[4:].customer_id, base.iloc[4:].device_id)
        pd.testing.assert_frame_equal(production.iloc[4:].reset_index(drop=True), continued)
        self.assertEqual(set(production.columns), set(DEVICE_GLOBAL_FEATURES))

    def test_one_hour_velocity_is_strictly_prior_and_matches_oracle(self):
        # Numeric timestamps are seconds.  The exact left boundary is included;
        # the t=3600 peers are mutually isolated until their entire batch ends.
        base = pd.DataFrame({"transaction_id": list("abcdefgh"),
                             "timestamp": [0, 1, 3599, 3600, 3600, 3601, 7200, 7201],
                             "customer_id": ["c", "c", "x", "c", "c", "c", "c", "c"],
                             "device_id": ["d", "x", "d", "d", "d", "d", "d", "d"],
                             "fraud": [0, 1, 0, 1, 0, 1, 0, 1]})
        oracle = velocity_oracle(base.timestamp, base.customer_id, base.device_id)
        production = velocity_production(base.timestamp, base.customer_id, base.device_id)
        pd.testing.assert_frame_equal(oracle, production)
        self.assertEqual(production.customer_prior_1h_count.tolist(), [0, 1, 0, 2, 2, 3, 3, 2])
        self.assertEqual(production.device_prior_1h_count.tolist(), [0, 0, 1, 2, 2, 3, 3, 2])
        # t=0 remains visible at t=3600; it expires only after the boundary.
        self.assertEqual(production.customer_prior_1h_count.iloc[3], 2)
        self.assertEqual(production.customer_prior_1h_count.iloc[7], 2)
        self.assertEqual(production.iloc[3].tolist(), production.iloc[4].tolist())
        # Equal-time row permutation cannot affect features attached to IDs.
        shuffled = pd.concat([base.iloc[:3], base.iloc[[4, 3]], base.iloc[5:]], ignore_index=True)
        shuffled_features = velocity_production(shuffled.timestamp, shuffled.customer_id, shuffled.device_id)
        pd.testing.assert_frame_equal(pd.concat([base[["transaction_id"]], production], axis=1).set_index("transaction_id").sort_index(),
                                      pd.concat([shuffled[["transaction_id"]], shuffled_features], axis=1).set_index("transaction_id").sort_index())
        # Future rows and labels are outside the velocity API and cannot alter prior output.
        future = pd.concat([base, base.iloc[[0]].assign(timestamp=9999, transaction_id="z")], ignore_index=True)
        pd.testing.assert_frame_equal(production, velocity_production(future.timestamp, future.customer_id, future.device_id).iloc[:len(base)].reset_index(drop=True))
        pd.testing.assert_frame_equal(production, velocity_production(base.timestamp, base.customer_id, base.device_id))
        # Splitting through tied batches preserves full-stream semantics.
        chunked = velocity_chunked([base.iloc[:4], base.iloc[4:6], base.iloc[6:]])
        pd.testing.assert_frame_equal(production, chunked)
        continued = velocity_from_prior_stream(base.iloc[:3].timestamp, base.iloc[:3].customer_id, base.iloc[:3].device_id,
                                                base.iloc[3:].timestamp, base.iloc[3:].customer_id, base.iloc[3:].device_id)
        pd.testing.assert_frame_equal(production.iloc[3:].reset_index(drop=True), continued)
        # Datetime nanoseconds and numeric seconds normalize to exactly the same window.
        datetime_features = velocity_production(pd.to_datetime(base.timestamp, unit="s"), base.customer_id, base.device_id)
        pd.testing.assert_frame_equal(production, datetime_features)
        self.assertEqual(set(production.columns), set(VELOCITY_FEATURES))

    def test_broad_velocity_windows_ratios_and_causal_contract(self):
        # Numeric inputs are seconds; this includes both exact 24h and 7d left boundaries.
        base = pd.DataFrame({"transaction_id": list("abcdefgh"),
                             "timestamp": [0, 3600, 86399, 86400, 86400, 604799, 604800, 604801],
                             "customer_id": ["c"] * 8, "device_id": ["d"] * 8,
                             "fraud": [0, 1, 0, 1, 0, 1, 0, 1]})
        oracle = broad_velocity_oracle(base.timestamp, base.customer_id, base.device_id)
        production = broad_velocity_production(base.timestamp, base.customer_id, base.device_id)
        pd.testing.assert_frame_equal(oracle, production)
        for prefix in ("customer", "device"):
            self.assertEqual(production[f"{prefix}_prior_1h_count"].tolist(), [0, 1, 0, 1, 1, 0, 1, 2])
            self.assertEqual(production[f"{prefix}_prior_24h_count"].tolist(), [0, 1, 2, 3, 3, 0, 1, 2])
            self.assertEqual(production[f"{prefix}_prior_7d_count"].tolist(), [0, 1, 2, 3, 3, 5, 6, 6])
            self.assertTrue(np.isfinite(production[f"{prefix}_1h_vs_7d_rate_ratio"]).all())
        # Same-timestamp peers observe the identical frozen state.
        pd.testing.assert_series_equal(production.iloc[3], production.iloc[4], check_names=False)
        shuffled = pd.concat([base.iloc[:3], base.iloc[[4, 3]], base.iloc[5:]], ignore_index=True)
        shuffled_features = broad_velocity_production(shuffled.timestamp, shuffled.customer_id, shuffled.device_id)
        pd.testing.assert_frame_equal(pd.concat([base[["transaction_id"]], production], axis=1).set_index("transaction_id").sort_index(),
                                      pd.concat([shuffled[["transaction_id"]], shuffled_features], axis=1).set_index("transaction_id").sort_index())
        future = pd.concat([base, base.iloc[[0]].assign(timestamp=9_999_999, transaction_id="z")], ignore_index=True)
        pd.testing.assert_frame_equal(production, broad_velocity_production(future.timestamp, future.customer_id, future.device_id).iloc[:len(base)].reset_index(drop=True))
        # Labels never enter the API; chunking through a tied batch and datetime nanoseconds agree exactly.
        pd.testing.assert_frame_equal(production, broad_velocity_chunked([base.iloc[:4], base.iloc[4:6], base.iloc[6:]]))
        pd.testing.assert_frame_equal(production, broad_velocity_production(pd.to_datetime(base.timestamp, unit="s"), base.customer_id, base.device_id))
        self.assertEqual(set(production.columns), set([*VELOCITY_FEATURES, *BROAD_FEATURES]))

    def test_r009_merchant_onehot_profile_is_narrow_and_target_independent(self):
        features = [*REACT2026_STATIC_FEATURES, *REACT2026_CUSTOMER_HISTORY_FEATURES,
                    *REACT2026_CUSTOMER_RELATIONSHIP_FEATURES, *REACT2026_DEVICE_GLOBAL_FEATURES,
                    *REACT2026_VELOCITY_FEATURES, "merchant_id"]
        config = replace(self.config, features=features,
                         categorical_features=["merchant_category", "device_type", "location", "payment_method", "transaction_type", "merchant_id"],
                         feature_profile=REACT2026_MERCHANT_ONEHOT_PROFILE,
                         model="catboost", model_params={"one_hot_max_size": 5000})
        config.validate()
        self.assertEqual(config.model_params["one_hot_max_size"], 5000)
        self.assertNotIn("customer_id", config.features)
        self.assertNotIn("device_id", config.features)
        self.assertNotIn("transaction_id", config.features)
        self.assertNotIn("fraud", config.features)
        forbidden = replace(config, features=[*features, "device_id"])
        with self.assertRaises(ValueError):
            forbidden.validate()

    def test_lightgbm_importance_uses_locked_onehot_order_without_introspection(self):
        config = replace(self.config, model="lightgbm", features=["number", "category"],
                         categorical_features=["category"])
        preprocessing = build_pipeline(config).named_steps["preprocess"]
        frame = pd.DataFrame({"number": [2., 1., 3.], "category": ["b", "a", "b"]})
        preprocessing.fit(frame)
        pipeline = SimpleNamespace(named_steps={"preprocess": preprocessing,
                                                 "model": SimpleNamespace(feature_importances_=np.array([3., 7., 11.]))})
        expected_order = ["number", "category_a", "category_b"]
        self.assertEqual(_lightgbm_transformed_feature_order(pipeline, config), expected_order)
        with patch.object(preprocessing, "get_feature_names_out", side_effect=AssertionError("unsupported introspection")):
            importance = _feature_importance(pipeline, config.features, config=config)
        self.assertEqual(len(importance), len(expected_order))
        self.assertEqual({row["feature"] for row in importance}, set(expected_order))
        self.assertEqual(importance[0], {"feature": "category_b", "importance": 11.0})

    def test_comparison_keeps_mock_and_live_metrics_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            (root / "outputs/reports").mkdir(parents=True)
            live = replace(self.config, metric="average_precision", metric_direction="higher", target="fraud",
                           validation_type="calendar_time", shuffle=False, time_column="time", calendar_folds=[
                               {"train_before": "2026-01-02", "valid_start": "2026-01-02", "valid_end": "2026-01-03"},
                               {"train_before": "2026-01-03", "valid_start": "2026-01-03", "valid_end": "2026-01-04"}])
            for experiment, config, metric in [("E999", self.config, .99), ("R001", live, .1)]:
                record = {"experiment_id": experiment, "config": config.to_dict(), "train_fingerprint": {"live": experiment.startswith("R")},
                          "split_signature": "live" if experiment.startswith("R") else "mock"}
                save_json(root / "outputs/reports" / f"{experiment}.json", record, exclusive=True)
            (root / "experiments/experiments.csv").write_text(
                "experiment_id,status,cv_mean,cv_std,training_seconds,model\nE999,completed,0.99,0,0,logistic\nR001,completed,0.1,0,0,logistic\n", encoding="utf-8")
            table, _ = comparison(root)
            self.assertEqual(table["experiment"].tolist(), ["R001"])

    def test_audit_ambiguity_sample_labels_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            empty = audit(root)
            self.assertIn("No competition files", empty)
            raw = root / "data/raw"
            raw.mkdir(parents=True)
            train = pd.DataFrame({"id": [1, 2, 3, 4], "x": [2, 2, 7, 8], "y": [0, 1, 1, 0]})
            train.to_csv(raw / "train.csv", index=False)
            pd.DataFrame({"id": [5, 6], "x": [2, 8]}).to_csv(raw / "test.csv", index=False)
            report = audit(root)
            self.assertIn("Target uncertain", report)
            pd.DataFrame({"id": [5, 6], "y": [0, 0]}).to_csv(raw / "sample_submission.csv", index=False)
            report = audit(root, full=True)
            self.assertIn("conflicting labels: 1", report)
            self.assertIn("No labels transferred", report)
            report = audit(root, max_rows=2)
            self.assertIn("SAMPLED", report)
            self.assertIn("lower bounds", report)
            self.assertGreater(len(list((root / "reports").glob("data_audit_*.md"))), 0)
            one, total = sampled_table(raw / "train.csv", 2, 42)
            two, _ = sampled_table(raw / "train.csv", 2, 42)
            pd.testing.assert_frame_equal(one, two)
            self.assertEqual(total, 4)

    def test_optional_models_fail_cleanly_without_installing(self):
        with patch("src.features.importlib.import_module", side_effect=ImportError("not installed")):
            for name in ("catboost", "lightgbm", "xgboost"):
                with self.assertRaisesRegex(ImportError, "Install explicitly"):
                    build_pipeline(replace(self.config, model=name))
        with self.assertRaisesRegex(ValueError, "no predict_proba"):
            build_pipeline(replace(self.config, model="tfidf_linear"))

    def test_ensemble_and_exported_notebook_replay(self):
        # A second fixture artifact copies the same fitted run. No second model is
        # selected or tuned; this exercises two-member orchestration only.
        fixture = copy.deepcopy(checked_record(self.root, "SMOKE001"))
        fixture["experiment_id"] = "SMOKE002"
        fixture.pop("submissions", None)
        fixture.pop("submission_path", None)
        save_json(self.root / "outputs/reports/SMOKE002.json", fixture, exclusive=True)
        with self.assertRaises(ValueError):
            blend_experiments(self.root, "SMOKE005", ["SMOKE001", "SMOKE002"], [.6, .6], "Bad weights", "Must reject")
        with contextlib.redirect_stdout(io.StringIO()):
            record = blend_experiments(self.root, "SMOKE003", ["SMOKE001", "SMOKE002"], [.5, .5], "Blend plumbing", "Explicit equal-weight fixture blend", method="mean", predict_test=True)
        self.assertAlmostEqual(record["cv_mean"], self.record["cv_mean"])
        generate_submission(self.root, "SMOKE003", description="blend_smoke")
        item = read_predictions(self.root / self.record["oof_path"])
        modified = (item[0], item[1], {**item[2], "class_order": ["yes", "no"]})
        with self.assertRaisesRegex(ValueError, "class_order"):
            compatible_artifacts([item, modified])
        exported = export_notebook(self.root, "SMOKE003")
        document = read_json(exported)
        rehearsal_path = os.environ.get("REACT_SMOKE_NOTEBOOK")
        if rehearsal_path:
            payload = {name: (self.root / "data/raw" / name).read_bytes().decode("utf-8") for name in ("train.csv", "test.csv", "sample_submission.csv")}
            setup = "import json\nfrom pathlib import Path\nimport tempfile\nDEMO_ROOT = Path(tempfile.mkdtemp(prefix='react-synthetic-rehearsal-', dir=globals().get('SMOKE_PARENT')))\nREPLAY_ROOT = DEMO_ROOT / 'replay'\nREPLAY_INPUTS = {}\nFIXTURES = json.loads(" + repr(json.dumps(payload)) + ")\nfor name, text in FIXTURES.items():\n    path = DEMO_ROOT / name\n    path.write_text(text, encoding='utf-8', newline='')\n    REPLAY_INPUTS['data/raw/' + name] = str(path)\nprint('Synthetic files and replay artifacts:', DEMO_ROOT)\n"
            rehearsal = notebook([markdown("# Synthetic Kaggle rehearsal — no competition data\n\nThis contains only 12 generated training rows and 3 generated inference rows. SMOKE002 duplicates the first fixture solely to test ensemble orchestration. Scores have no competition meaning. Run all cells privately and Save Version to verify runtime access and source portability. Do not upload these CSVs to a competition."), code(setup), *document["cells"]])
            save_json(Path(rehearsal_path), rehearsal, exclusive=True)
            document = rehearsal
        replay_root = self.root / "notebook_replay"
        namespace = {"REPLAY_ROOT": replay_root, "REPLAY_INPUTS": {str(Path("data/raw") / name).replace('\\', '/'): str(self.root / "data/raw" / name) for name in ("train.csv", "test.csv", "sample_submission.csv")}}
        namespace["SMOKE_PARENT"] = str(self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in document["cells"]:
                if cell["cell_type"] == "code":
                    text = "".join(cell["source"])
                    exec(compile(text, "<exported-notebook>", "exec"), namespace)
        replay_root = namespace["REPLAY_ROOT"]
        replay = checked_record(replay_root, "SMOKE003")
        self.assertEqual(sha256(self.root / record["prediction_path"]), sha256(replay_root / replay["prediction_path"]))
        self.assertEqual(len(list((replay_root / "submissions").glob("*.csv"))), 2)


if __name__ == "__main__":
    unittest.main()
