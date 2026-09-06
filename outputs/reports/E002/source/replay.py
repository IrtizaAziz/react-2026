"""Cross-environment replay checks for immutable experiment provenance."""
from pathlib import Path
import numpy as np
import pandas as pd

from .metrics import score
from .predict import read_predictions
from .utils import read_json, sha256


DEFAULT_TOLERANCES = {"rtol": 1e-7, "atol": 1e-9, "metric_rtol": 1e-7, "metric_atol": 1e-9}


def _numeric_comparison(expected, actual, *, rtol, atol):
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    difference = np.abs(actual - expected)
    relative = np.divide(difference, np.abs(expected), out=np.full_like(difference, np.nan), where=expected != 0)
    close = np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    return {"max_absolute_difference": float(np.nanmax(difference)) if difference.size else 0.0,
            "max_relative_difference": float(np.nanmax(relative)) if np.isfinite(relative).any() else None,
            "values_outside_tolerance": int((~close).sum())}, bool(close.all())


def compare_predictions(expected_path, actual_path, *, rtol, atol):
    """Check prediction semantics without concealing differing artifact hashes."""
    expected_frame, expected_values, expected_meta = read_predictions(expected_path)
    actual_frame, actual_values, actual_meta = read_predictions(actual_path)
    report = {"expected_sha256": sha256(expected_path), "actual_sha256": sha256(actual_path),
              "hash_match": sha256(expected_path) == sha256(actual_path),
              "shape_match": np.shape(expected_values) == np.shape(actual_values),
              "identity_match": expected_meta["identity_hash"] == actual_meta["identity_hash"],
              "class_order_match": expected_meta["class_order"] == actual_meta["class_order"],
              "prediction_columns_match": expected_meta["prediction_columns"] == actual_meta["prediction_columns"],
              "prediction_kind_match": expected_meta["prediction_kind"] == actual_meta["prediction_kind"]}
    if not all(report[key] for key in ("shape_match", "identity_match", "class_order_match", "prediction_columns_match", "prediction_kind_match")):
        report.update(reproduced=False, reason="Prediction shape, identity, class mapping, or columns differ")
        return report
    if expected_meta["prediction_kind"] == "label":
        report.update(values_outside_tolerance=int(np.any(expected_values != actual_values)), reproduced=bool(np.array_equal(expected_values, actual_values)))
        return report
    numeric, reproduced = _numeric_comparison(expected_values, actual_values, rtol=rtol, atol=atol)
    report.update(numeric, reproduced=reproduced)
    return report


def compare_submission(expected_path, actual_path, *, id_columns, rtol, atol):
    """Require submission row/ID alignment and numerically equivalent values."""
    expected, actual = pd.read_csv(expected_path), pd.read_csv(actual_path)
    value_columns = [column for column in expected if column not in id_columns]
    report = {"expected_sha256": sha256(expected_path), "actual_sha256": sha256(actual_path),
              "hash_match": sha256(expected_path) == sha256(actual_path),
              "columns_match": expected.columns.tolist() == actual.columns.tolist(),
              "shape_match": expected.shape == actual.shape}
    if not report["columns_match"] or not report["shape_match"]:
        report.update(reproduced=False, reason="Submission columns or shape differ")
        return report
    report["identity_match"] = expected[id_columns].astype("string").equals(actual[id_columns].astype("string"))
    if not report["identity_match"]:
        report.update(reproduced=False, reason="Submission row/ID alignment differs")
        return report
    numeric, reproduced = _numeric_comparison(expected[value_columns].to_numpy(), actual[value_columns].to_numpy(), rtol=rtol, atol=atol)
    report.update(numeric, reproduced=reproduced)
    return report


def replay_oof_score(expected_record, actual_record, root, config, *, rtol, atol):
    """Recompute OOF metric and compare it with the recorded CV/OOF score."""
    frame, predictions, _ = read_predictions(Path(root) / actual_record["oof_path"])
    training = pd.read_csv(config.path(root, config.train_file))
    if frame["__row__"].tolist() != list(range(len(training))):
        raise ValueError("OOF row alignment differs from training data")
    value = score(training[config.target], predictions, config)
    recorded = float(expected_record["cv_mean"])
    return {"recorded_cv_mean": recorded, "replayed_oof_score": value,
            "absolute_difference": abs(value - recorded), "within_tolerance": bool(np.isclose(value, recorded, rtol=rtol, atol=atol))}
