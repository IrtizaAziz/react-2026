"""Cross-environment replay checks for immutable experiment provenance."""
from pathlib import Path
import numpy as np
import pandas as pd

from .metrics import score
from .data import read_table
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
    """Replay temporal OOF metrics without treating pooled AP as mean fold AP."""
    frame, predictions, metadata = read_predictions(Path(root) / actual_record["oof_path"])
    training = read_table(config.path(root, config.train_file), id_columns=config.id_columns)
    if frame["__row__"].tolist() != list(range(len(training))):
        raise ValueError("OOF row alignment differs from training data")
    for column in config.id_columns:
        if column not in frame or frame[column].astype("string").tolist() != training[column].astype("string").tolist():
            raise ValueError("OOF ID alignment differs from training data")
    if "__fold__" not in frame:
        raise ValueError("OOF artifact lacks fold assignments")
    folds = frame["__fold__"].to_numpy(dtype=int)
    covered = folds >= 0
    if not covered.any() or metadata.get("split_signature") != actual_record.get("split_signature"):
        raise ValueError("OOF coverage or split signature differs from experiment provenance")
    raw = np.asarray(predictions)
    finite = np.isfinite(raw.astype(float)) if config.prediction_kind != "label" else pd.notna(raw)
    row_finite = finite if finite.ndim == 1 else finite.all(axis=1)
    if row_finite[covered].sum() != covered.sum() or row_finite[~covered].any():
        raise ValueError("OOF predictions must be finite exactly on validated rows; warmup rows remain unpredicted")
    replayed_folds = []
    for fold in range(len(actual_record["fold_scores"])):
        mask = folds == fold
        if not mask.any():
            raise ValueError(f"OOF artifact has no rows for fold {fold}")
        replayed_folds.append(score(training.loc[mask, config.target], raw[mask], config))
    recorded_folds = np.asarray(expected_record["fold_scores"], dtype=float)
    replayed_folds = np.asarray(replayed_folds, dtype=float)
    if recorded_folds.shape != replayed_folds.shape:
        raise ValueError("Recorded and replayed fold counts differ")
    fold_match = np.isclose(recorded_folds, replayed_folds, rtol=rtol, atol=atol)
    recorded_mean = float(expected_record["cv_mean"])
    replayed_mean = float(replayed_folds.mean())
    pooled = score(training.loc[covered, config.target], raw[covered], config)
    return {"recorded_cv_mean": recorded_mean, "replayed_cv_mean": replayed_mean,
            "recorded_fold_scores": recorded_folds.tolist(), "replayed_fold_scores": replayed_folds.tolist(),
            "folds_within_tolerance": bool(fold_match.all()),
            "cv_mean_within_tolerance": bool(np.isclose(recorded_mean, replayed_mean, rtol=rtol, atol=atol)),
            "pooled_covered_oof_score": pooled,
            "within_tolerance": bool(fold_match.all() and np.isclose(recorded_mean, replayed_mean, rtol=rtol, atol=atol))}


def replay_fold_score(path, root, config, *, expected_fold, split_signature):
    """Replay one saved validation-fold artifact before the next fold starts."""
    frame, predictions, metadata = read_predictions(path)
    training = read_table(config.path(root, config.train_file), id_columns=config.id_columns)
    rows = frame["__row__"].to_numpy(dtype=int)
    if len(rows) == 0 or len(np.unique(rows)) != len(rows) or rows.min() < 0 or rows.max() >= len(training):
        raise ValueError("Fold prediction rows are invalid")
    if "__fold__" not in frame or not np.array_equal(frame["__fold__"].to_numpy(dtype=int), np.full(len(frame), expected_fold)):
        raise ValueError("Fold prediction assignment differs from the expected validation fold")
    if metadata.get("split_signature") != split_signature:
        raise ValueError("Fold prediction split signature differs from the experiment")
    for column in config.id_columns:
        if frame[column].astype("string").tolist() != training.iloc[rows][column].astype("string").tolist():
            raise ValueError("Fold prediction ID alignment differs from training data")
    value = score(training.iloc[rows][config.target], predictions, config)
    return {"fold": expected_fold, "rows": int(len(rows)), "average_precision": float(value),
            "prediction_identity_hash": metadata["identity_hash"], "verified": True}
