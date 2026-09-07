"""Saved-pipeline inference and prediction artifacts with explicit identity metadata."""
if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from .config import Config, ROOT
from .data import load_test
from .utils import checked_record, finish_record, object_hash, read_json, save_frame, save_json, sha256


def model_predictions(pipeline, features, config):
    if config.prediction_kind == "probability":
        if not hasattr(pipeline, "predict_proba"):
            raise ValueError("Model does not provide probabilities")
        predictions = pipeline.predict_proba(features)
        classes = list(pipeline.classes_)
        if set(classes) != set(range(len(config.class_order))):
            raise ValueError("Fitted model classes differ from the configured class mapping")
        return predictions[:, [classes.index(i) for i in range(len(config.class_order))]]
    if config.prediction_kind == "decision":
        if not hasattr(pipeline, "decision_function"):
            raise ValueError("Model does not provide decision scores")
        return pipeline.decision_function(features)
    predictions = pipeline.predict(features)
    if config.task == "classification":
        return np.asarray(config.class_order)[np.asarray(predictions, dtype=int)]
    return predictions


def save_predictions(path, frame, predictions, config, fingerprint, *, folds=None, split_signature=None, row_numbers=None,
                     allow_uncovered_nulls=False):
    predictions = np.asarray(predictions)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if len(predictions) != len(frame):
        raise ValueError("Prediction row count mismatch")
    if allow_uncovered_nulls or (folds is not None and not np.isfinite(predictions).all()):
        if folds is None:
            raise ValueError("Null OOF predictions require canonical fold assignments")
        finite = np.isfinite(predictions)
        covered = finite if predictions.ndim == 1 else finite.all(axis=1)
        missing = ~finite if predictions.ndim == 1 else ~finite.any(axis=1)
        if not np.array_equal(covered | missing, np.ones(len(predictions), dtype=bool)) or not np.array_equal(covered, np.asarray(folds) >= 0):
            raise ValueError("OOF nulls must occur exactly on uncovered canonical rows")
    elif not np.isfinite(predictions).all():
        raise ValueError("Prediction artifacts require finite outputs unless canonical OOF warmup nulls are enabled")
    if row_numbers is None:
        row_numbers = np.arange(len(frame))
    row_numbers = np.asarray(row_numbers)
    if row_numbers.shape != (len(frame),) or len(np.unique(row_numbers)) != len(row_numbers):
        raise ValueError("Prediction row numbers must be unique and aligned with the frame")
    result = pd.DataFrame({"__row__": row_numbers})
    for column in config.id_columns:
        result[column] = frame[column].reset_index(drop=True).astype("string")
    identity = object_hash(result.to_dict(orient="list"))
    if folds is not None:
        result["__fold__"] = folds
    columns = [f"pred_{i}" for i in range(predictions.shape[1])]
    for i, column in enumerate(columns):
        result[column] = predictions[:, i]
    save_frame(path, result)
    metadata = {"prediction_kind": config.prediction_kind, "class_order": config.class_order,
                "prediction_columns": columns, "id_columns": config.id_columns,
                "data_fingerprint": fingerprint, "identity_hash": identity,
                "split_signature": split_signature, "sha256": sha256(path)}
    save_json(Path(path).with_suffix(".meta.json"), metadata, exclusive=True)
    return metadata


def read_predictions(path):
    path = Path(path)
    metadata = read_json(path.with_suffix(".meta.json"))
    if sha256(path) != metadata["sha256"]:
        raise ValueError(f"Prediction artifact integrity check failed: {path.name}")
    dtype = {column: "string" for column in metadata["id_columns"]}
    if metadata["prediction_kind"] == "label" and all(isinstance(c, str) for c in metadata["class_order"]):
        dtype.update({column: "string" for column in metadata["prediction_columns"]})
    frame = pd.read_csv(path, dtype=dtype)
    identity = frame[["__row__", *metadata["id_columns"]]]
    if object_hash(identity.to_dict(orient="list")) != metadata["identity_hash"]:
        raise ValueError("Prediction identity metadata mismatch")
    predictions = frame[metadata["prediction_columns"]].to_numpy()
    if predictions.shape[1] == 1:
        predictions = predictions[:, 0]
    return frame, predictions, metadata


def predict_experiment(root, experiment, *, test_file=None, aggregation=None):
    root = Path(root)
    record = checked_record(root, experiment)
    config = Config(**record["config"])
    if record.get("ensemble"):
        raise ValueError("Infer ensemble members first, then blend under a new experiment ID")
    if aggregation != "mean" or config.prediction_kind == "label":
        raise ValueError("Inference requires explicit aggregation='mean' and numeric predictions")
    if record.get("prediction_path"):
        raise FileExistsError("Predictions already exist; they will not be overwritten")
    if test_file:
        config.test_file = test_file
    frame, fingerprint = load_test(config, root)
    accumulated = None
    for path in record["model_paths"]:
        model_path = root / path
        if sha256(model_path) != record["model_hashes"][path]:
            raise ValueError("Saved model integrity check failed")
        with model_path.open("rb") as handle:
            pipeline = pickle.load(handle)  # Only artifacts generated by this trusted repository.
        predictions = model_predictions(pipeline, frame[config.features], config)
        accumulated = predictions.astype(float) if accumulated is None else accumulated + predictions
    predictions = accumulated / len(record["model_paths"])
    if not np.isfinite(predictions).all():
        raise ValueError("Inference generated non-finite predictions")
    path = f"outputs/predictions/{experiment}.csv"
    save_predictions(root / path, frame, predictions, config, fingerprint)
    record.update(prediction_path=path, test_fingerprint=fingerprint,
                  inference={"test_file": config.test_file, "aggregation": aggregation})
    finish_record(root, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--test-file")
    parser.add_argument("--aggregation", required=True, choices=["mean"])
    args = parser.parse_args()
    try:
        result = predict_experiment(args.root, args.experiment, test_file=args.test_file, aggregation=args.aggregation)
        print(result["prediction_path"])
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Inference stopped: {exc}\n")


if __name__ == "__main__":
    main()
