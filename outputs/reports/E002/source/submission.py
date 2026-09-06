"""Validate and generate local CSVs only. There is no Kaggle upload function."""
if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .config import Config, ROOT, load_config
from .data import read_table, require_columns
from .metrics import labels_from_probabilities
from .predict import read_predictions
from .utils import checked_record, finish_record, save_frame, save_json, sha256

SUBMISSION_FIELDS = ("sample_file", "submission_columns", "submission_kind", "submission_alignment", "positive_class", "label_threshold")


def validate_probabilities(values, *, sums=False):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be finite and in [0, 1]; no automatic correction")
    if sums and (values.ndim != 2 or not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Multiclass probability rows must sum to 1 within 1e-6")


def build_submission(sample, artifact, predictions, metadata, config):
    if len(sample) != len(artifact) or len(predictions) != len(sample):
        raise ValueError("Sample/prediction row count mismatch")
    if not config.submission_columns or not config.submission_kind:
        raise ValueError("Configure submission_columns and submission_kind from Evaluation")
    expected = [c for c in sample.columns if c not in config.id_columns]
    if expected != config.submission_columns:
        raise ValueError("Configured prediction columns/order do not exactly match sample submission")
    if any(str(c).lower().startswith("unnamed:") for c in sample):
        raise ValueError("Sample contains an apparent accidental index column; inspect the official format")
    if metadata["prediction_kind"] != config.prediction_kind or metadata["class_order"] != config.class_order or metadata["id_columns"] != config.id_columns:
        raise ValueError("Prediction metadata is incompatible with the submission configuration")
    if config.submission_alignment == "id":
        if not config.id_columns:
            raise ValueError("ID alignment requires explicit id_columns")
        for frame in (sample, artifact):
            require_columns(frame, config.id_columns, "Submission IDs")
            if frame[config.id_columns].isna().any().any() or frame.duplicated(config.id_columns).any():
                raise ValueError("Submission IDs must be unique and nonmissing")
        def keys(frame):
            return list(frame[config.id_columns].astype("string").itertuples(index=False, name=None))
        sample_keys, prediction_keys = keys(sample), keys(artifact)
        if set(sample_keys) != set(prediction_keys):
            raise ValueError("Sample and predictions have different IDs")
        positions = {key: i for i, key in enumerate(prediction_keys)}
        predictions = np.asarray(predictions)[[positions[key] for key in sample_keys]]
    elif config.submission_alignment == "position":
        if config.id_columns:
            raise ValueError("Use ID alignment when id_columns are configured")
        if not np.array_equal(artifact["__row__"].to_numpy(), np.arange(len(sample))):
            raise ValueError("Positional predictions must retain original test row order")
    else:
        raise ValueError("Choose submission_alignment='id' or explicitly 'position'")
    values = np.asarray(predictions)
    if config.prediction_kind == "probability":
        validate_probabilities(values, sums=True)
    if config.submission_kind == "label":
        if config.prediction_kind == "probability":
            values = labels_from_probabilities(values, config)
        elif config.prediction_kind != "label":
            raise ValueError("Label output needs labels or an explicit probability decision rule")
        if len(config.submission_columns) != 1 or not set(values.ravel()) <= set(config.class_order):
            raise ValueError("Label output must contain one column of configured class labels")
    elif config.submission_kind == "probability":
        if config.prediction_kind != "probability":
            raise ValueError("Probability output requires probability predictions")
        if len(config.submission_columns) == 1:
            if len(config.class_order) != 2 or config.positive_class not in config.class_order:
                raise ValueError("Single probability column requires binary classes and explicit positive_class")
            values = values[:, config.class_order.index(config.positive_class)]
        elif len(config.submission_columns) != len(config.class_order):
            raise ValueError("Probability columns must follow the complete configured class_order")
    elif config.submission_kind == "value":
        if config.prediction_kind != "value":
            raise ValueError("Value output requires regression-value predictions")
    else:
        raise ValueError("submission_kind must be value, label, or probability")
    if values.ndim == 1:
        values = values[:, None]
    if values.shape != (len(sample), len(config.submission_columns)):
        raise ValueError("Prediction shape does not match the official template")
    if pd.isna(values).any():
        raise ValueError("Submission predictions contain missing values")
    if config.submission_kind != "label" and not np.isfinite(values.astype(float)).all():
        raise ValueError("Submission predictions contain non-finite values")
    result = sample.copy()
    for i, column in enumerate(config.submission_columns):
        result[column] = values[:, i]
    if list(result.columns) != list(sample.columns) or result.isna().any().any():
        raise ValueError("Submission schema changed or contains missing values")
    return result


def generate_submission(root, experiment, *, config=None, description="baseline"):
    root = Path(root)
    record = checked_record(root, experiment)
    resolved = Config(**record["config"])
    if config:
        for key in SUBMISSION_FIELDS:
            setattr(resolved, key, getattr(config, key))
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", description):
        raise ValueError("Description must contain only letters, digits, underscores, or hyphens")
    if not record.get("prediction_path"):
        raise ValueError("Generate test predictions explicitly before creating a submission")
    sample_path = resolved.path(root, resolved.sample_file)
    sample = read_table(sample_path, id_columns=resolved.id_columns)
    artifact, predictions, metadata = read_predictions(root / record["prediction_path"])
    if metadata["data_fingerprint"] != record["test_fingerprint"]:
        raise ValueError("Prediction data fingerprint disagrees with experiment provenance")
    result = build_submission(sample, artifact, predictions, metadata, resolved)
    score_text = f"{record['cv_mean']:.6f}".replace("-", "m").replace(".", "p")
    filename = f"sub_{experiment}_{record['model']}_{description}_cv{score_text}.csv"
    path = root / "submissions" / filename
    save_frame(path, result)
    submission_record = {"experiment_id": experiment, "submission_path": f"submissions/{filename}",
                         "sha256": sha256(path), "sample_sha256": sha256(sample_path),
                         "prediction_sha256": metadata["sha256"], "git_commit": record["git_commit"],
                         "config": {key: getattr(resolved, key) for key in SUBMISSION_FIELDS},
                         "kaggle_notebook_version": None, "uploaded": False}
    save_json(path.with_suffix(".json"), submission_record, exclusive=True)
    record.setdefault("submissions", []).append(submission_record)
    record["submission_path"] = submission_record["submission_path"]
    finish_record(root, record)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", type=Path, help="Override only submission fields using a complete config JSON")
    parser.add_argument("--description", default="baseline")
    args = parser.parse_args()
    try:
        print(generate_submission(args.root, args.experiment, config=load_config(args.config) if args.config else None, description=args.description))
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Submission stopped: {exc}\n")


if __name__ == "__main__":
    main()
