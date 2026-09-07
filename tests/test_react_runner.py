"""Read-only preflight and immutable execution tests for the REACT runner."""
import hashlib
import inspect
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import patch

from src.config import ROOT, load_config
from src.react_runner import (_blend_preflight, _next_id, _require_clean_execution_config,
                              _run_argv, _training_preflight, preflight, run, supervise,
                              wait_for_artifact)
from src.feature_experiment import (FeatureSpec, POLICY, _certify, _load_parent_frame, _module, _module_hashes,
                                    cache_parent, preflight_spec)
from src.utils import object_hash, sha256


class ReactRunnerTests(unittest.TestCase):
    def test_cache_parent_rejects_missing_or_incomplete_run(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "Missing immutable record"):
                cache_parent(temporary, "R027")
            report = Path(temporary) / "outputs/reports/R026.json"; report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"status": "running"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not completed"):
                cache_parent(temporary, "R026")

    def _fake_feature_module(self, directory, name="temporary_feature_module"):
        source = Path(directory) / f"{name}.py"; source.write_text("# deterministic test module\n", encoding="utf-8")
        module = ModuleType(name); module.__file__ = str(source); module.AVAILABLE_FEATURES = ("first", "second")
        module.build_features = lambda frame, requested: __import__("pandas").DataFrame(
            {name: range(len(frame)) for name in requested}, index=frame.index)
        module.certification_cases = lambda: {key: (lambda: None) for key in (
            "oracle", "strict_past", "equal_timestamp_isolation", "permutation_invariance",
            "duplicate_same_timestamp_pair", "future_independence", "label_independence",
            "chunk_vs_whole", "module_specific")}
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        return module

    def test_feature_spec_subset_contract_and_rejections(self):
        with TemporaryDirectory() as temporary:
            module = self._fake_feature_module(temporary)
            spec = FeatureSpec("R999", "R017", module.__name__, ("second",), "A real hypothesis", predict_test=False)
            loaded, available = _module(spec)
            self.assertEqual(loaded, module); self.assertEqual(available, ("first", "second"))
            self.assertEqual(list(module.build_features(__import__("pandas").DataFrame({"first": [1], "second": [2]}), ["second"]).columns), ["second"])
            with self.assertRaisesRegex(ValueError, "subset"):
                _module(FeatureSpec("R999", "R017", module.__name__, ("missing",), "h", predict_test=False))
            with self.assertRaisesRegex(ValueError, "test inference"):
                FeatureSpec("R999", "R017", module.__name__, ("first",), "h", predict_test=True).validate()
            referenced = FeatureSpec("R999", "R017", module.__name__, ("first",), "h", reference_runs=("R025",))
            self.assertEqual(referenced.normalized()["reference_runs"], ["R025"])
            with self.assertRaisesRegex(ValueError, "reference_runs"):
                FeatureSpec("R999", "R017", module.__name__, ("first",), "h", reference_runs=("R025", "R025")).validate()

    def test_certification_reuse_and_changed_source_requires_new_record(self):
        with TemporaryDirectory() as temporary:
            module = self._fake_feature_module(temporary); hashes = _module_hashes(module)
            first, reused = _certify(temporary, module, module.AVAILABLE_FEATURES, hashes)
            self.assertFalse(reused); self.assertTrue(all(first["results"][name] for name in first["results"]))
            second, reused = _certify(temporary, module, module.AVAILABLE_FEATURES, hashes)
            self.assertTrue(reused); self.assertEqual(first, second)
            Path(module.__file__).write_text("# changed source\n", encoding="utf-8")
            changed, reused = _certify(temporary, module, module.AVAILABLE_FEATURES, _module_hashes(module))
            self.assertFalse(reused); self.assertNotEqual(first["identity"], changed["identity"])

    def test_feature_preflight_rejects_r030_style_missing_case_before_ready(self):
        with TemporaryDirectory() as temporary:
            module = self._fake_feature_module(temporary, "missing_case_module")
            module.certification_cases = lambda: {key: (lambda: None) for key in (
                "strict_past", "equal_timestamp_isolation", "permutation_invariance",
                "future_independence", "label_independence", "chunk_vs_whole")}
            spec = FeatureSpec("R030", "R026", module.__name__, ("first",), "A real hypothesis", predict_test=False)
            parent = {"status": "completed", "provenance_path": "outputs/reports/R026/"}
            with patch("src.feature_experiment.FeatureSpec.load", return_value=spec), \
                 patch("src.feature_experiment._record", side_effect=[parent, parent]), \
                 patch("src.feature_experiment.load_config"), \
                 patch("src.feature_experiment._module", return_value=(module, module.AVAILABLE_FEATURES)), \
                 patch("src.feature_experiment._module_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "oracle, duplicate_same_timestamp_pair"):
                    preflight_spec(temporary, Path(temporary) / "r030.json")

    def test_feature_preflight_rejects_undeclared_derived_column_before_ready(self):
        with TemporaryDirectory() as temporary:
            module = self._fake_feature_module(temporary, "derived_column_module")
            module.build_features = lambda frame, requested: frame.loc[:, ["undeclared_derived_column"]]
            spec = FeatureSpec("R031", "R026", module.__name__, ("first",), "A real hypothesis", predict_test=False)
            parent = {"status": "completed", "provenance_path": "outputs/reports/R026/"}
            with patch("src.feature_experiment.FeatureSpec.load", return_value=spec), \
                 patch("src.feature_experiment._record", side_effect=[parent, parent]), \
                 patch("src.feature_experiment.load_config"), \
                 patch("src.feature_experiment._module", return_value=(module, module.AVAILABLE_FEATURES)), \
                 patch("src.feature_experiment._module_hashes", return_value={}):
                with self.assertRaisesRegex(KeyError, "undeclared_derived_column"):
                    preflight_spec(temporary, Path(temporary) / "r031.json")

    def test_feature_policy_preserves_thresholds_and_distinct_gate_semantics(self):
        self.assertEqual(POLICY["screen_maximum_drops_vs_incumbent"], {"F2": .010, "F2_late_corrected": .005, "July_1_15": .005})
        self.assertEqual(POLICY["submission_gates_vs_incumbent"]["July_1_15"], .003)
        self.assertEqual(POLICY["classification_vs_parent"]["win_f2"], .002)

    def test_verified_parent_matrix_cache_and_id_mismatch_rejection(self):
        import pandas as pd
        with TemporaryDirectory() as temporary:
            root = Path(temporary); cache = root / "outputs/feature_matrices/R900.pkl"; cache.parent.mkdir(parents=True)
            frame = pd.DataFrame({"transaction_id": pd.Series(["a", "b"], dtype="string"), "fraud": [0, 1], "parent_feature": [1., 2.]})
            frame.to_pickle(cache)
            config_path = root / "outputs/reports/R900/config.json"; config_path.parent.mkdir(parents=True); config_path.write_text("{}", encoding="utf-8")
            source = config_path.parent / "source"; source.mkdir(); frozen = source / "data.py"; frozen.write_text("# frozen\n", encoding="utf-8")
            source_hashes = {"data.py": sha256(frozen)}
            fingerprint = {"sha256": "raw", "rows": 2, "columns": list(frame.columns)}
            manifest = {"parent": "R900", "parent_config_sha256": sha256(config_path), "train_fingerprint": fingerprint,
                        "schema_version": 2, "split_signature": "split", "feature_columns": ["parent_feature"], "columns": list(frame.columns),
                        "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()}, "row_count": 2,
                        "matrix_sha256": sha256(cache), "index_hash": object_hash(frame.index.to_list()),
                        "id_hash": object_hash(frame[["transaction_id"]].astype("string").to_dict(orient="list")),
                        "label_hash": object_hash(frame["fraud"].to_list()),
                        "frozen_source_provenance": {"snapshot_path": str(source.relative_to(root)), "source_hashes": source_hashes}}
            cache.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
            config = SimpleNamespace(id_columns=["transaction_id"], target="fraud", features=["parent_feature"])
            record = {"provenance_path": "outputs/reports/R900/", "train_fingerprint": fingerprint, "split_signature": "split", "source_hashes": source_hashes}
            loaded, loaded_fingerprint, details = _load_parent_frame(root, "R900", config, record)
            self.assertTrue(details["used"]); self.assertEqual(loaded_fingerprint, fingerprint); pd.testing.assert_frame_equal(loaded, frame)
            frozen.write_text("# changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen source snapshot"):
                _load_parent_frame(root, "R900", config, record)
            frozen.write_text("# frozen\n", encoding="utf-8")
            manifest["id_hash"] = "wrong"; cache.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ID or label"):
                _load_parent_frame(root, "R900", config, record)
    def test_training_preflight_is_read_only(self):
        root = Path(ROOT); ledger = root / "experiments/experiments.csv"; before = hashlib.sha256(ledger.read_bytes()).hexdigest()
        result = _training_preflight(root, root / "config_r017.json", "R013", [root / "tests/test_r017_merchant_history.py"])
        self.assertEqual(result["split_signature"], "6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47")
        self.assertEqual(before, hashlib.sha256(ledger.read_bytes()).hexdigest())

    def test_blend_preflight_is_read_only(self):
        root = Path(ROOT); ledger = root / "experiments/experiments.csv"; before = hashlib.sha256(ledger.read_bytes()).hexdigest()
        result = _blend_preflight(root, "R013", ["R013", "R011"], [.75, .25])
        self.assertFalse(result["weight_search"]); self.assertEqual(before, hashlib.sha256(ledger.read_bytes()).hexdigest())

    def test_explicit_run_uses_preflight_without_token(self):
        root = Path(ROOT); payload = preflight(root, "blend", parent="R013", members=["R013", "R011"], weights=[.75, .25])
        self.assertNotIn("authorization_token", payload)
        self.assertEqual(list(inspect.signature(run).parameters), ["root", "payload", "hypothesis", "change", "experiment"])
        with self.assertRaisesRegex(ValueError, "experiment ID"):
            run(root, payload, "x", "x", "R999")
        self.assertEqual(payload["experiment_id"], _next_id(root))

    def test_rejects_missing_tests_test_inference_and_wrong_id(self):
        root = Path(ROOT)
        with self.assertRaisesRegex(ValueError, "causal/parity"):
            _training_preflight(root, root / "config_r017.json", "R013", [])
        config = load_config(root / "config_r017.json"); config.predict_test = True; config.aggregation = "mean"
        with self.assertRaisesRegex(ValueError, "never permits"):
            _require_clean_execution_config(config)
        payload = preflight(root, "blend", parent="R013", members=["R013", "R011"], weights=[.75, .25])
        with self.assertRaisesRegex(ValueError, "experiment ID"):
            run(root, payload, "x", "x", "R999")

    def test_supervise_reports_heartbeat_and_completed_record(self):
        class Process:
            pid = 321
            def __init__(self): self.results = iter((None, 0, 0))
            def poll(self): return next(self.results)
        with TemporaryDirectory() as temporary:
            root = Path(temporary); report = root / "outputs/reports/R999.json"; report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"status": "completed", "fold_scores": [.7, .8], "provenance_path": "outputs/reports/R999/"}))
            output, calls, clock = [], [], iter((0, 60))
            code = supervise(root, ["run", "blend"], "R999", heartbeat_seconds=60,
                             popen=lambda command, cwd: calls.append((command, cwd)) or Process(),
                             clock=lambda: next(clock), sleep=lambda _: None, emit=output.append)
        self.assertEqual(code, 0); self.assertEqual(calls[0][0][2:], ["src.react_runner", "run", "blend"])
        self.assertTrue(any("heartbeat" in line for line in output)); self.assertIn("status=completed", output[-1]); self.assertIn("F2=0.800000", output[-1])

    def test_supervise_preserves_failure_and_does_not_require_a_record(self):
        class Process:
            pid = 654
            def poll(self): return 7
        output = []
        with TemporaryDirectory() as temporary:
            self.assertEqual(supervise(temporary, ["run", "blend"], "R998", popen=lambda *_args, **_kwargs: Process(), emit=output.append), 7)
        self.assertIn("exit_code=7 status=missing-report", output[-1])
        with self.assertRaisesRegex(ValueError, "at least one second"):
            supervise(temporary, [], "R998", heartbeat_seconds=.5)

    def test_supervise_recreates_the_exact_run_arguments(self):
        args = SimpleNamespace(root=Path("repo"), mode="training", parent="R017", config="config_r999.json",
                               tests=["tests/test_r999.py"], members=None, weights=None, experiment="R999",
                               hypothesis="h", change="c")
        self.assertEqual(_run_argv(args), ["--root", "repo", "run", "training", "--parent", "R017", "--config", "config_r999.json", "--tests", "tests/test_r999.py", "--experiment", "R999", "--hypothesis", "h", "--change", "c"])

    def test_wait_for_existing_json_and_non_json_artifacts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); report = root / "outputs/reports/R999/post_fit_report.json"; report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"experiment_id": "R999", "decision": "WIN", "fold_scores": [.7, .8]}))
            output = []
            self.assertEqual(wait_for_artifact(root, report.relative_to(root), emit=output.append), 0)
            self.assertIn("experiment_id=R999", output[-1]); self.assertIn("F2=0.800000", output[-1])
            text = root / "outputs/reports/R999/done.txt"; text.write_text("ready")
            self.assertEqual(wait_for_artifact(root, text.relative_to(root), emit=output.append), 0)
            self.assertEqual(output[-1], "REACT wait-for ready: path=outputs\\reports\\R999\\done.txt")

    def test_wait_for_polls_then_times_out_without_writing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); path = root / "outputs/reports/R998/post_fit_report.json"; output = []
            clock = iter((0, 0, 25))
            def appear(_): path.parent.mkdir(parents=True); path.write_text("{}")
            self.assertEqual(wait_for_artifact(root, path.relative_to(root), poll_seconds=25, timeout_seconds=30,
                                               clock=lambda: next(clock), sleep=appear, emit=output.append), 0)
            self.assertIn("ready", output[-1])
            self.assertEqual(wait_for_artifact(root, "missing.json", timeout_seconds=0, emit=output.append), 1)
            self.assertIn("timeout", output[-1])
            with self.assertRaisesRegex(ValueError, "within"):
                wait_for_artifact(root, "../outside.json")


if __name__ == "__main__": unittest.main()
