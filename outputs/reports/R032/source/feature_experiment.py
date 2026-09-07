"""Spec-driven, feature-only REACT experiments.

This module deliberately has no test-inference or submission path.  It is an
additive adapter around the normal immutable trainer, not a replacement for
the historical R runners.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, ROOT, load_config
from .predict import read_predictions
from .utils import object_hash, read_json, save_json, sha256


POLICY = {
    "version": "react-feature-experiment-v1",
    "screen_maximum_drops_vs_incumbent": {"F2": 0.010, "F2_late_corrected": 0.005, "July_1_15": 0.005},
    "submission_gates_vs_incumbent": {"July_1_15": 0.003, "F2_late_corrected": 0.002,
                                        "F2": 0.003, "F1": -0.002,
                                        "July_1_7": -0.002, "July_8_15": -0.002},
    "classification_vs_parent": {"win_f2": 0.002, "lose_recent": -0.002},
}
WINDOWS = (
    ("F2_early", 1, "2026-05-15", "2026-06-15"),
    ("F2_late_corrected", 1, "2026-06-15", "2026-07-16"),
    ("July_1_15", 1, "2026-07-01", "2026-07-16"),
    ("June_15_30", 1, "2026-06-15", "2026-07-01"),
    ("July_1_7", 1, "2026-07-01", "2026-07-08"),
    ("July_8_15", 1, "2026-07-08", "2026-07-16"),
)
REQUIRED_CASES = ("oracle", "strict_past", "equal_timestamp_isolation", "permutation_invariance",
                  "duplicate_same_timestamp_pair", "future_independence", "label_independence", "chunk_vs_whole")


@dataclass(frozen=True)
class FeatureSpec:
    experiment_id: str
    parent: str
    feature_module: str
    added_features: tuple[str, ...]
    hypothesis: str
    incumbent: str = "R017"
    predict_test: bool = False
    reference_runs: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: Path) -> "FeatureSpec":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {"experiment_id", "parent", "feature_module", "added_features", "hypothesis", "incumbent", "predict_test", "reference_runs"}
        unknown = set(payload) - allowed
        missing = {"experiment_id", "parent", "feature_module", "added_features", "hypothesis", "predict_test"} - set(payload)
        if unknown or missing:
            raise ValueError(f"Feature spec fields invalid; unknown={sorted(unknown)}, missing={sorted(missing)}")
        spec = cls(**{**payload, "added_features": tuple(payload["added_features"]), "reference_runs": tuple(payload.get("reference_runs", ()))})
        spec.validate()
        return spec

    def validate(self):
        if not re.fullmatch(r"R\d{3,}", self.experiment_id):
            raise ValueError("Feature spec experiment_id must be R###")
        if not all(isinstance(value, str) and value for value in (self.parent, self.incumbent, self.feature_module)):
            raise ValueError("parent, incumbent, and feature_module must be nonempty strings")
        if not self.hypothesis.strip():
            raise ValueError("Feature spec requires a human-written hypothesis")
        if self.predict_test is not False:
            raise ValueError("Feature runner permanently rejects test inference")
        if not self.added_features or any(not isinstance(item, str) or not item for item in self.added_features) or len(set(self.added_features)) != len(self.added_features):
            raise ValueError("added_features must be a nonempty ordered unique list")
        if (any(not isinstance(item, str) or not re.fullmatch(r"R\d{3,}", item) for item in self.reference_runs) or
                len(set(self.reference_runs)) != len(self.reference_runs)):
            raise ValueError("reference_runs must be an ordered unique list of R### IDs")

    def normalized(self):
        return {**asdict(self), "added_features": list(self.added_features), "reference_runs": list(self.reference_runs), "policy_version": POLICY["version"]}


def _record(root, experiment):
    path = Path(root) / "outputs/reports" / f"{experiment}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing immutable record: {experiment}")
    result = read_json(path)
    if result.get("status") != "completed":
        raise ValueError(f"{experiment} is not completed")
    return result


def _next_id(root):
    from .react_runner import _next_id as next_id
    return next_id(root)


def _module(spec):
    module = importlib.import_module(spec.feature_module)
    available = tuple(getattr(module, "AVAILABLE_FEATURES", ()))
    if not available or len(set(available)) != len(available):
        raise ValueError("Feature module must declare unique AVAILABLE_FEATURES")
    if not set(spec.added_features) <= set(available):
        raise ValueError("added_features is not a subset of AVAILABLE_FEATURES")
    if not callable(getattr(module, "build_features", None)) or not callable(getattr(module, "certification_cases", None)):
        raise ValueError("Feature module requires build_features and certification_cases")
    return module, available


def _module_hashes(module):
    files = [Path(module.__file__)]
    for dependency in getattr(module, "SOURCE_DEPENDENCIES", ()): files.append(Path(dependency))
    if any(not path.exists() for path in files): raise FileNotFoundError("Feature module dependency is missing")
    return {str(path.resolve()): sha256(path) for path in sorted(set(files))}


def _certification_path(root, module_name, identity):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", module_name)
    return Path(root) / "outputs/certifications" / slug / f"{identity}.json"


def _validate_feature_output(values, raw_frame, requested_features):
    if not isinstance(values, pd.DataFrame) or list(values.columns) != list(requested_features):
        raise ValueError("Feature module must return exactly requested columns in requested order")
    if len(values) != len(raw_frame) or not values.index.equals(raw_frame.index):
        raise ValueError("Feature module output row/index alignment mismatch")


def _raw_compatibility_frame():
    """Minimal runner-schema frame; feature modules may not require parent features."""
    return pd.DataFrame({"transaction_id": ["a", "b", "c", "d"], "timestamp": [0, 1, 2, 2],
                         "customer_id": ["u", "u", "v", "w"], "device_id": ["d", "d", "d", "d"],
                         "merchant_id": ["m", "m", "n", "n"], "location": ["l", "l", "l", "x"]},
                        index=pd.Index([17, 4, 29, 8], name="raw_row"))


def _validate_raw_frame_compatibility(module, available):
    raw = _raw_compatibility_frame()
    values = module.build_features(raw, list(available))
    _validate_feature_output(values, raw, available)


def _run_certification_cases(module, available):
    """Validate and execute every causal check before an experiment is ready."""
    cases = module.certification_cases()
    if not isinstance(cases, dict):
        raise ValueError("Feature certification is missing required causal cases")
    missing = tuple(name for name in REQUIRED_CASES if name not in cases)
    if missing:
        raise ValueError(f"Feature certification is missing required causal cases: {', '.join(missing)}")
    results = {}
    for name, check in cases.items():
        if not callable(check):
            raise ValueError(f"Certification case {name} is not callable")
        check(); results[name] = True
    if not all(results.get(name) is True for name in REQUIRED_CASES):
        raise ValueError("Feature certification did not pass required causal cases")
    _validate_raw_frame_compatibility(module, available)
    results["raw_frame_compatibility"] = True
    return results


def _certify(root, module, available, module_hashes, *, force=False):
    identity = object_hash({"module_hashes": module_hashes, "available_features": list(available), "policy": POLICY["version"]})
    path = _certification_path(root, module.__name__, identity)
    if path.exists() and not force:
        record = read_json(path)
        if record.get("identity") != identity or not all(record.get("results", {}).get(name) is True for name in REQUIRED_CASES):
            raise ValueError("Stored feature certification is invalid")
        return record, True
    results = _run_certification_cases(module, available)
    record = {"schema_version": 1, "identity": identity, "module": module.__name__, "module_source_sha256": module_hashes,
              "AVAILABLE_FEATURES": list(available), "results": results, "runner_policy_version": POLICY["version"]}
    if path.exists():
        if read_json(path) != record: raise ValueError("Existing certification differs for the same immutable identity")
    else: save_json(path, record, exclusive=True)
    return record, False


def _cache_paths(root, parent):
    cache = Path(root) / "outputs/feature_matrices" / f"{parent}.pkl"
    return cache, cache.with_suffix(".json")


def _frozen_source_provenance(root, parent, parent_record):
    """Verify the immutable source snapshot used to reproduce a parent matrix."""
    source = Path(root) / parent_record["provenance_path"] / "source"
    expected = parent_record.get("source_hashes", {})
    if not expected or not source.is_dir():
        raise ValueError("Parent frozen source snapshot is missing")
    observed = {path.name: sha256(path) for path in source.iterdir() if path.is_file()}
    if observed != expected:
        raise ValueError("Parent frozen source snapshot differs from immutable provenance")
    return {"snapshot_path": str(source.relative_to(root)), "source_hashes": expected}


def _frame_manifest(root, parent, parent_config, parent_record, frame):
    provenance = _frozen_source_provenance(root, parent, parent_record)
    config_path = Path(root) / parent_record["provenance_path"] / "config.json"
    feature_columns = list(parent_config.features)
    if not feature_columns or any(name not in frame for name in feature_columns):
        raise ValueError("Frozen parent matrix does not contain its exact feature manifest")
    if parent == "R017" and len(feature_columns) != 50:
        raise ValueError("Frozen R017 matrix does not contain its exact 50-feature manifest")
    if len(frame) != parent_record["train_fingerprint"]["rows"]:
        raise ValueError("Frozen parent matrix row count differs from immutable parent")
    ids = frame[parent_config.id_columns].astype("string").to_dict(orient="list")
    return {"schema_version": 2, "experiment_id": parent, "creation_method": "frozen_source_snapshot",
            "parent": parent, "parent_config_sha256": sha256(config_path),
            "train_fingerprint": parent_record["train_fingerprint"], "split_signature": parent_record["split_signature"],
            "feature_columns": feature_columns, "columns": list(frame.columns),
            "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()}, "row_count": len(frame),
            "index_hash": object_hash(frame.index.to_list()), "id_hash": object_hash(ids),
            "label_hash": object_hash(frame[parent_config.target].to_list()),
            "frozen_source_provenance": provenance,
            "spec_provenance": {"report_path": str((Path(root) / parent_record["provenance_path"]).relative_to(root)),
                                 "spec_sha256": sha256(Path(root) / parent_record["provenance_path"] / "decision_report.json")
                                 if (Path(root) / parent_record["provenance_path"] / "decision_report.json").exists() else None}}


def _validate_cached_parent_frame(root, parent, parent_config, parent_record, cache_path, manifest_path):
    manifest = read_json(manifest_path)
    required = {"schema_version", "parent", "parent_config_sha256", "train_fingerprint", "split_signature",
                "feature_columns", "columns", "dtypes", "row_count", "matrix_sha256", "index_hash", "id_hash",
                "label_hash", "frozen_source_provenance"}
    if not required <= set(manifest):
        raise ValueError("Parent matrix cache manifest is incomplete")
    if manifest["schema_version"] != 2 or sha256(cache_path) != manifest["matrix_sha256"]:
        raise ValueError("Parent matrix cache hash or schema mismatch")
    config_path = Path(root) / parent_record["provenance_path"] / "config.json"
    if (manifest["parent"] != parent or manifest["parent_config_sha256"] != sha256(config_path) or
            manifest["train_fingerprint"] != parent_record.get("train_fingerprint") or
            manifest["split_signature"] != parent_record.get("split_signature") or
            manifest["feature_columns"] != list(parent_config.features) or
            manifest["frozen_source_provenance"] != _frozen_source_provenance(root, parent, parent_record)):
        raise ValueError("Parent matrix cache provenance mismatch")
    frame = pd.read_pickle(cache_path)
    if (list(frame.columns) != manifest["columns"] or
            {name: str(dtype) for name, dtype in frame.dtypes.items()} != manifest["dtypes"] or
            len(frame) != manifest["row_count"] or object_hash(frame.index.to_list()) != manifest["index_hash"]):
        raise ValueError("Parent matrix cache schema, dtype, or row-order mismatch")
    ids = frame[parent_config.id_columns].astype("string").to_dict(orient="list")
    if object_hash(ids) != manifest["id_hash"] or object_hash(frame[parent_config.target].to_list()) != manifest["label_hash"]:
        raise ValueError("Parent matrix cache ID or label mismatch")
    return frame, manifest


def _load_frozen_parent_frame(root, parent, parent_config_path, parent_record):
    """Build a frame with a temporary package containing only frozen parent source."""
    source = Path(root) / parent_record["provenance_path"] / "source"
    _frozen_source_provenance(root, parent, parent_record)
    with tempfile.TemporaryDirectory(prefix=f"react_{parent}_frozen_") as temporary:
        package = f"_react_frozen_{parent.lower()}_{uuid.uuid4().hex}"
        package_dir = Path(temporary) / package
        shutil.copytree(source, package_dir)
        spec = importlib.util.spec_from_file_location(package, package_dir / "__init__.py", submodule_search_locations=[str(package_dir)])
        module = importlib.util.module_from_spec(spec); sys.modules[package] = module
        try:
            spec.loader.exec_module(module)
            frozen_config = importlib.import_module(f"{package}.config")
            frozen_data = importlib.import_module(f"{package}.data")
            config = frozen_config.load_config(parent_config_path)
            frame, fingerprint = frozen_data.load_training(config, root)
        finally:
            for name in [name for name in sys.modules if name == package or name.startswith(f"{package}.")]:
                sys.modules.pop(name, None)
    return frame, fingerprint


def materialize_parent_matrix_cache(root, parent):
    """Create or reuse a verified generic parent cache without training or reservation."""
    root = Path(root); parent_record = _record(root, parent)
    parent_config_path = root / parent_record["provenance_path"] / "config.json"
    parent_config = load_config(parent_config_path); cache_path, manifest_path = _cache_paths(root, parent)
    if cache_path.exists() or manifest_path.exists():
        if not cache_path.exists() or not manifest_path.exists():
            raise ValueError("Parent matrix cache and manifest must both exist")
        frame, manifest = _validate_cached_parent_frame(root, parent, parent_config, parent_record, cache_path, manifest_path)
        return frame, parent_record["train_fingerprint"], {"used": True, "path": str(cache_path.relative_to(root)),
                "matrix_sha256": manifest["matrix_sha256"], "manifest_sha256": sha256(manifest_path)}
    # Feature experiments record a parent plus an immutable feature spec; their
    # final matrix is the frozen parent matrix with the frozen module appended.
    normalized = parent_record.get("feature_experiment_provenance", {}).get("normalized_spec")
    if normalized:
        base = normalized["parent"]
        base_record = _record(root, base)
        base_config_path = root / base_record["provenance_path"] / "config.json"
        base_config = load_config(base_config_path)
        frame, fingerprint, _ = materialize_parent_matrix_cache(root, base)
        source = Path(root) / parent_record["provenance_path"] / "source"
        with tempfile.TemporaryDirectory(prefix=f"react_{parent}_module_") as temporary:
            package = f"_react_frozen_{parent.lower()}_{uuid.uuid4().hex}"; package_dir = Path(temporary) / package
            shutil.copytree(source, package_dir)
            module_spec = importlib.util.spec_from_file_location(package, package_dir / "__init__.py", submodule_search_locations=[str(package_dir)])
            package_module = importlib.util.module_from_spec(module_spec); sys.modules[package] = package_module
            try:
                module_spec.loader.exec_module(package_module)
                module = importlib.import_module(f"{package}.merchant_new_customers")
                raw = pd.read_csv(parent_config.path(root, parent_config.train_file), dtype={column: "string" for column in parent_config.id_columns})
                values = module.build_features(raw, normalized["added_features"])
            finally:
                for name in [name for name in sys.modules if name == package or name.startswith(f"{package}.")]: sys.modules.pop(name, None)
        if len(values) != len(frame) or not values.index.equals(raw.index): raise ValueError("Frozen feature module row alignment mismatch")
        for name in normalized["added_features"]: frame[name] = values[name].to_numpy(copy=False)
    else:
        frame, fingerprint = _load_frozen_parent_frame(root, parent, parent_config_path, parent_record)
    if fingerprint != parent_record.get("train_fingerprint"):
        raise ValueError("Frozen parent training input fingerprint differs from immutable record")
    split = read_json(parent_config.path(root, parent_config.splits_file))
    if split.get("signature") != parent_record.get("split_signature"):
        raise ValueError("Frozen parent split signature differs from immutable record")
    manifest = _frame_manifest(root, parent, parent_config, parent_record, frame)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".pkl", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        frame.to_pickle(temporary_path); manifest["matrix_sha256"] = sha256(temporary_path)
        if cache_path.exists() or manifest_path.exists():
            raise ValueError("Parent matrix cache appeared during materialization")
        os.replace(temporary_path, cache_path)
        save_json(manifest_path, manifest, exclusive=True)
    finally:
        if temporary_path.exists(): temporary_path.unlink()
    return _load_parent_frame(root, parent, parent_config, parent_record)


def cache_parent(root, parent, *, verify_only=False):
    """Materialize or validate one completed immutable experiment's train matrix."""
    root = Path(root); cache_path, manifest_path = _cache_paths(root, parent)
    if verify_only:
        record = _record(root, parent)
        config = load_config(root / record["provenance_path"] / "config.json")
        if not cache_path.exists() or not manifest_path.exists():
            raise ValueError(f"Verified cache is missing for {parent}")
        frame, manifest = _validate_cached_parent_frame(root, parent, config, record, cache_path, manifest_path)
        return {"experiment_id": parent, "verified": True, "path": str(cache_path.relative_to(root)),
                "matrix_sha256": manifest["matrix_sha256"], "manifest_sha256": sha256(manifest_path),
                "rows": len(frame), "features": len(config.features)}
    frame, fingerprint, details = materialize_parent_matrix_cache(root, parent)
    return {"experiment_id": parent, "verified": True, "path": details["path"],
            "matrix_sha256": details["matrix_sha256"], "manifest_sha256": details["manifest_sha256"],
            "rows": len(frame), "features": len(frame.columns)}


def _load_parent_frame(root, parent, parent_config, parent_record):
    """Load only a hash-verified parent matrix cache, otherwise rebuild safely."""
    from .data import load_training
    cache_path, manifest_path = _cache_paths(root, parent)
    if cache_path.exists() or manifest_path.exists():
        if not cache_path.exists() or not manifest_path.exists():
            raise ValueError("Parent matrix cache and manifest must both exist")
        frame, manifest = _validate_cached_parent_frame(root, parent, parent_config, parent_record, cache_path, manifest_path)
        return frame, parent_record["train_fingerprint"], {"used": True, "path": str(cache_path.relative_to(root)), "matrix_sha256": manifest["matrix_sha256"], "manifest_sha256": sha256(manifest_path)}
    # Rebuilding through the checkout is safe only when it is byte-identical
    # to the frozen parent implementation.  Otherwise a verified cache is
    # required; silently regenerating changed historical features is forbidden.
    frozen = parent_record.get("source_hashes", {})
    for name in ("data.py", "config.py", "customer_history.py", "customer_relationships.py", "velocity.py"):
        if name in frozen and sha256(Path(__file__).with_name(name)) != frozen[name]:
            raise ValueError("No verified parent matrix cache and frozen parent source differs from this checkout")
    train_path = parent_config.path(root, parent_config.train_file)
    # Open/read first: Windows sharing failures must occur before reservation.
    with train_path.open("rb") as handle: handle.read(1)
    frame, fingerprint = load_training(parent_config, root)
    if fingerprint != parent_record.get("train_fingerprint"):
        raise ValueError("Parent training input fingerprint differs from immutable record")
    return frame, fingerprint, {"used": False, "reason": "no verified parent matrix cache available"}


def _build_candidate(root, spec, parent_config, parent_record, module):
    parent_frame, fingerprint, cache = _load_parent_frame(root, spec.parent, parent_config, parent_record)
    raw = pd.read_csv(parent_config.path(root, parent_config.train_file), dtype={column: "string" for column in parent_config.id_columns})
    values = module.build_features(raw, list(spec.added_features))
    _validate_feature_output(values, raw, spec.added_features)
    if len(values) != len(parent_frame):
        raise ValueError("Feature module output row alignment mismatch with parent")
    candidate = parent_frame.copy()
    for name in spec.added_features: candidate[name] = values[name].to_numpy(copy=False)
    if candidate.columns.duplicated().any() or candidate[list(parent_config.features)].equals(parent_frame[list(parent_config.features)]) is False:
        raise ValueError("Parent feature parity failed")
    if not candidate.index.equals(parent_frame.index) or not candidate[parent_config.target].equals(parent_frame[parent_config.target]):
        raise ValueError("Parent row or label parity failed")
    for column in parent_config.id_columns:
        if not candidate[column].astype("string").equals(parent_frame[column].astype("string")):
            raise ValueError("Parent transaction-ID alignment failed")
    return candidate, fingerprint, cache


def preflight_spec(root, spec_path):
    root = Path(root); spec = FeatureSpec.load(spec_path); parent = _record(root, spec.parent); incumbent = _record(root, spec.incumbent)
    parent_config_path = root / parent["provenance_path"] / "config.json"
    parent_config = load_config(parent_config_path)
    module, available = _module(spec); hashes = _module_hashes(module)
    # Do not report a spec ready until the complete causal contract has passed.
    _run_certification_cases(module, available)
    # Use the actual raw training schema too, before the caller can authorize a run.
    raw_contract = pd.read_csv(parent_config.path(root, parent_config.train_file), nrows=8,
                               dtype={column: "string" for column in parent_config.id_columns})
    _validate_feature_output(module.build_features(raw_contract, list(spec.added_features)), raw_contract,
                             spec.added_features)
    train_path = parent_config.path(root, parent_config.train_file)
    with train_path.open("rb") as handle: handle.read(1)
    train_hash = sha256(train_path)
    if train_hash != parent["train_fingerprint"]["sha256"]: raise ValueError("train.csv SHA256 differs from parent")
    split_path = parent_config.path(root, parent_config.splits_file)
    split = read_json(split_path)
    if split.get("signature") != parent.get("split_signature"): raise ValueError("Locked split signature differs from parent")
    # Validate parent prerequisites before a supervised child can be launched.
    # This remains read-only and deliberately rejects changed-checkout fallback.
    _load_parent_frame(root, spec.parent, parent_config, parent)
    if spec.incumbent != spec.parent:
        incumbent_config = load_config(root / incumbent["provenance_path"] / "config.json")
        _load_parent_frame(root, spec.incumbent, incumbent_config, incumbent)
    if spec.experiment_id != _next_id(root): raise ValueError("Spec experiment ID is not the next unused immutable ID")
    identity = object_hash({"module_hashes": hashes, "available_features": list(available), "policy": POLICY["version"]})
    cert = _certification_path(root, module.__name__, identity)
    return {"mode": "feature", "spec": spec.normalized(), "spec_sha256": object_hash(spec.normalized()), "parent": spec.parent,
            "incumbent": spec.incumbent, "feature_manifest": [*parent_config.features, *spec.added_features],
            "feature_delta": list(spec.added_features), "model_recipe": {"inherited_from": spec.parent, "model": parent_config.model,
            "model_params": parent_config.model_params, "categorical_features": parent_config.categorical_features,
            "seed": parent_config.seed, "supervised_weighting": parent_config.supervised_weighting},
            "train_fingerprint": parent["train_fingerprint"], "train_sha256": train_hash, "split_signature": split["signature"],
            "split_sha256": sha256(split_path), "parent_config_sha256": sha256(parent_config_path), "module_source_sha256": hashes,
            "certification_path": str(cert.relative_to(root)), "certification_reused": cert.exists(), "test_inference_disabled": True}


def _scores(root, record, frame, config):
    from sklearn.metrics import average_precision_score
    oof, predictions, _ = read_predictions(Path(root) / record["oof_path"])
    if not oof["__row__"].eq(pd.RangeIndex(len(frame))).all(): raise ValueError("OOF row alignment failed")
    for column in config.id_columns:
        if not oof[column].astype("string").equals(frame[column].astype("string")): raise ValueError("OOF transaction-ID alignment failed")
    pred = predictions[:, 1] if np.asarray(predictions).ndim == 2 else predictions
    folds, times = oof["__fold__"].to_numpy(), frame[config.time_column].astype("string").to_numpy()
    result = {"F2": float(average_precision_score(frame.loc[folds == 1, config.target], pred[folds == 1]))}
    f1 = folds == 0
    if f1.any() and np.isfinite(np.asarray(pred)[f1]).all():
        result["F1"] = float(average_precision_score(frame.loc[f1, config.target], pred[f1]))
    for name, fold, start, end in WINDOWS:
        mask = (folds == fold) & (times >= start) & (times < end)
        if np.isfinite(np.asarray(pred)[mask]).all():
            result[name] = float(average_precision_score(frame.loc[mask, config.target], pred[mask]))
    return result


def _reference_scores(root, references, parent_record):
    """Best-effort descriptive comparisons; never influence a decision."""
    aliases = {"F1": "F1", "F2": "F2", "F2 early": "F2_early", "F2_early": "F2_early",
               "corrected F2 late": "F2_late_corrected", "F2_late_corrected": "F2_late_corrected",
               "July": "July_1_15", "July 1-15": "July_1_15", "June 15-30": "June_15_30",
               "July 1-7": "July_1_7", "July 8-15": "July_8_15"}
    def resolve(experiment):
        base = Path(root) / "outputs/reports" / experiment
        candidates = [base / "decision_report.json"]
        if base.is_dir():
            candidates += [p for p in sorted(base.glob("*.json")) if p.name != "decision_report.json"]
        candidates.append(Path(root) / "outputs/reports" / f"{experiment}.json")
        for path in candidates:
            if not path.exists():
                continue
            payload = read_json(path)
            if not isinstance(payload, dict):
                continue
            raw = payload.get("metrics") or payload.get("scores") or {}
            metrics = {aliases[key]: value for key, value in raw.items() if key in aliases and isinstance(value, (int, float)) and not isinstance(value, bool)}
            if metrics:
                return metrics
        return {}
    result = {}
    for experiment in references:
        try:
            record = _record(root, experiment)
            if record.get("split_signature") != parent_record.get("split_signature"):
                result[experiment] = {"compatible": False, "reason": "split signature differs"}; continue
            metrics = resolve(experiment)
            if not metrics:
                result[experiment] = {"compatible": False, "reason": "decision metrics are unavailable"}; continue
            result[experiment] = {"compatible": True, "metrics": metrics}
        except (FileNotFoundError, ValueError, KeyError) as exc:
            result[experiment] = {"compatible": False, "reason": str(exc)}
    return result


def _run_diagnostics(root, module, candidate, record, config, added_features):
    hook = getattr(module, "diagnostics", None)
    if hook is None: return None, None
    try:
        if not callable(hook): raise ValueError("feature module diagnostics must be callable when declared")
        oof, _, _ = read_predictions(Path(root) / record["oof_path"])
        payload = hook(candidate.copy(), oof.copy(), config=config, added_features=tuple(added_features))
        # The JSON serializer is the interface validation and keeps this artifact deterministic.
        path = Path(root) / record["provenance_path"] / "feature_diagnostics.json"
        save_json(path, payload, exclusive=True)
        return str(path.relative_to(root)), None
    except (OSError, TypeError, ValueError) as exc:
        return None, str(exc)


def execute_spec(root, spec_path, authorize, *, recertify=False):
    from .train import train_experiment
    from .validation import make_splits
    root = Path(root); payload = preflight_spec(root, spec_path); spec = FeatureSpec.load(spec_path)
    if authorize != spec.experiment_id or authorize != _next_id(root): raise ValueError("--authorize must equal the spec and next immutable R-ID")
    module, available = _module(spec); hashes = _module_hashes(module)
    certification, reused = _certify(root, module, available, hashes, force=recertify)
    # Re-run every external check and materialize the complete candidate before reservation.
    payload = preflight_spec(root, spec_path)
    parent_record = _record(root, spec.parent); parent_config = load_config(root / parent_record["provenance_path"] / "config.json")
    candidate, fingerprint, cache = _build_candidate(root, spec, parent_config, parent_record, module)
    pairs, assignment, splits = make_splits(candidate, parent_config, fingerprint, reuse=parent_config.path(root, parent_config.splits_file))
    if splits["signature"] != parent_record["split_signature"]: raise ValueError("Candidate split signature differs from parent")
    incumbent_record = _record(root, spec.incumbent)
    incumbent_frame, _, _ = _load_parent_frame(root, spec.incumbent, load_config(root / incumbent_record["provenance_path"] / "config.json"), incumbent_record)
    incumbent_scores = _scores(root, incumbent_record, incumbent_frame, load_config(root / incumbent_record["provenance_path"] / "config.json"))
    config = copy.deepcopy(parent_config); config.feature_profile = None; config.features = [*parent_config.features, *spec.added_features]
    config.execution_fold_order = [1, 0]
    config.diagnostic_windows = [{"name": name, "fold": fold, "start": start, "end": end} for name, fold, start, end in WINDOWS]
    config.early_stop_screen = {"fold": 1, "references": {"F2": incumbent_scores["F2"], "F2_late_corrected": incumbent_scores["F2_late_corrected"], "July_1_15": incumbent_scores["July_1_15"]}, "maximum_drops": POLICY["screen_maximum_drops_vs_incumbent"]}
    change = f"Append {list(spec.added_features)} from {spec.feature_module} to immutable {spec.parent}; inherit {spec.parent}'s complete model recipe."
    provenance = {"normalized_spec": spec.normalized(), "normalized_spec_sha256": payload["spec_sha256"],
                  "spec_path": str(Path(spec_path).resolve()), "spec_sha256": sha256(spec_path), "module_source_sha256": hashes,
                  "runner_source_sha256": {"feature_experiment.py": sha256(Path(__file__)),
                                           "react_runner.py": sha256(Path(__file__).with_name("react_runner.py"))},
                  "parent_config_sha256": payload["parent_config_sha256"],
                  "split_sha256": payload["split_sha256"], "split_signature": splits["signature"], "train_fingerprint": fingerprint,
                  "parent_matrix_cache": cache, "certification": certification, "certification_reused": reused, "policy": POLICY}
    # Final exact-ID check is intentionally the last operation before reserve_experiment inside train_experiment.
    if authorize != spec.experiment_id or authorize != _next_id(root): raise ValueError("Experiment ID changed before reservation")
    record = train_experiment(config, spec.experiment_id, spec.hypothesis, change, root=root, parent=spec.parent,
                              prepared_frame=candidate, prepared_fingerprint=fingerprint, provenance_extra=provenance)
    candidate_scores = _scores(root, record, candidate, config)
    parent_scores = _scores(root, parent_record, _load_parent_frame(root, spec.parent, parent_config, parent_record)[0], parent_config)
    parent_delta = {name: candidate_scores[name] - parent_scores[name] for name in candidate_scores if name in parent_scores}
    incumbent_delta = {name: candidate_scores[name] - incumbent_scores[name] for name in candidate_scores if name in incumbent_scores}
    recent = [parent_delta.get(name, 0) for name in ("F2_late_corrected", "July_1_15", "July_1_7", "July_8_15")]
    classification = "WIN" if parent_delta.get("F2", -np.inf) >= .002 and all(value >= 0 for value in recent) else "LOSE" if any(value < -.002 for value in recent) else "FLAT-MIXED"
    gates = {name: incumbent_delta.get(name, -np.inf) >= value for name, value in POLICY["submission_gates_vs_incumbent"].items()}
    references = _reference_scores(root, spec.reference_runs, parent_record)
    for experiment, details in references.items():
        if details.get("compatible"):
            details["delta_vs_reference"] = {name: candidate_scores[name] - value for name, value in details["metrics"].items() if name in candidate_scores}
    diagnostics_path, diagnostics_error = _run_diagnostics(root, module, candidate, record, config, spec.added_features)
    report = {"experiment_id": spec.experiment_id, "parent": spec.parent, "incumbent": spec.incumbent, "metrics": candidate_scores,
              "delta_vs_parent": parent_delta, "delta_vs_incumbent": incumbent_delta, "classification": classification,
              "submission_gate_pass": bool(all(gates.values()) and not record.get("early_stopped")), "submission_gates": gates,
              "early_stop_policy": POLICY["screen_maximum_drops_vs_incumbent"], "early_stopped": bool(record.get("early_stopped")),
              "certification_reused": reused, "feature_importances": record.get("feature_importances", {}),
              "fold_replays": record.get("fold_replays", {}), "runtime_seconds": record.get("training_seconds"),
              "reference_runs": references, "diagnostics_path": diagnostics_path, "diagnostics_error": diagnostics_error,
              "test_inference": False, "submission_created": False}
    save_json(root / record["provenance_path"] / "decision_report.json", report, exclusive=True)
    from .experiment_docs import write_experiment_card
    write_experiment_card(root, record, report)
    return report
