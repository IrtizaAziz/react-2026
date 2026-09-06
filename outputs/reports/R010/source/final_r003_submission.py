"""One immutable all-train R003 refit and causal test-inference package.

This is deliberately separate from the immutable R003 validation record.  It
uses its frozen configuration but never averages its fold models or mutates its
OOF/CV artifacts.
"""
if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
import base64
from datetime import datetime, timezone
import pickle
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .config import Config, ROOT, load_config
from .customer_history import customer_history_from_prior_stream
from .data import add_feature_profile, load_training, read_table, require_columns, validate_feature_profile
from .features import build_pipeline
from .predict import model_predictions, save_predictions
from .submission import build_submission
from .utils import dumps, environment, read_json, save_frame, save_json, sha256, snapshot_source


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _causal_test_features(train, raw_test, config):
    """Create test features in original test order from train-initialized state."""
    if config.target in raw_test:
        raise ValueError("Test input contains a target column")
    require_columns(raw_test, ["timestamp", "customer_id", "amount_bdt", *config.id_columns], "Causal test inputs")
    train_times = pd.to_datetime(train["timestamp"], errors="raise")
    test_times = pd.to_datetime(raw_test["timestamp"], errors="raise")
    if train_times.max() >= test_times.min():
        raise ValueError("Test stream must begin strictly after the complete labeled train stream")
    order = np.argsort(test_times.to_numpy(dtype="datetime64[ns]"), kind="stable")
    chronological = raw_test.iloc[order]
    start = time.perf_counter()
    history = customer_history_from_prior_stream(
        train["timestamp"], train["customer_id"], train["amount_bdt"],
        chronological["timestamp"], chronological["customer_id"], chronological["amount_bdt"],
    )
    history.index = chronological.index
    history = history.reindex(raw_test.index)
    result = pd.concat([add_feature_profile(raw_test, "react2026_static"), history], axis=1)
    validate_feature_profile(result, config)
    if list(result.index) != list(raw_test.index) or len(result) != len(raw_test):
        raise ValueError("Causal test features lost original test-row alignment")
    checks = {
        "train_before_test": True,
        "test_processed_chronologically": True,
        "test_timestamp_tie_count": int(test_times.duplicated(keep=False).sum()),
        "test_timestamp_batches": int(test_times.nunique()),
        "test_history_generation_seconds": time.perf_counter() - start,
        "history_uses_raw_fields_only": ["timestamp", "customer_id", "amount_bdt"],
        "same_timestamp_batch_update": "predict/query entire batch before updating raw state",
        "test_row_order_restored": True,
    }
    return result, checks


def _export_notebook(package, config, validated_r003, output):
    """Create a self-contained local notebook ready for a committed Kaggle run."""
    source = {path.name: base64.b64encode(path.read_bytes()).decode("ascii")
              for path in sorted((package / "source").glob("*.py"))}
    settings = config.to_dict()
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# REACT 2026 — R003 final causal refit\n",
            "This notebook reconstructs the frozen R003 final-refit package. Map only the organizer-provided files, run it, inspect the generated CSV, then commit a Kaggle version. It does not upload a submission.\n",
        ]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "import base64, json, subprocess, sys\n",
            "from pathlib import Path\n",
            "RUN_ROOT = Path('/kaggle/working/react-r003-final').resolve()\n",
            "INPUTS = {\n",
            "    'train': '/kaggle/input/REPLACE/train.csv',\n",
            "    'test': '/kaggle/input/REPLACE/test.csv',\n",
            "    'sample': '/kaggle/input/REPLACE/sample_submission.csv',\n",
            "}\n",
            "SOURCES = " + repr(source) + "\n",
            "CONFIG = " + repr(settings) + "\n",
            "VALIDATED_R003_REPORT = " + repr(validated_r003) + "\n",
            "if any('REPLACE' in path for path in INPUTS.values()):\n",
            "    raise ValueError('Set INPUTS to the organizer-provided Kaggle file paths first')\n",
            "source_dir = RUN_ROOT / 'src'\n",
            "source_dir.mkdir(parents=True, exist_ok=False)\n",
            "for name, encoded in SOURCES.items():\n",
            "    (source_dir / name).write_bytes(base64.b64decode(encoded))\n",
            "report_path = RUN_ROOT / 'outputs' / 'reports' / 'R003.json'\n",
            "report_path.parent.mkdir(parents=True, exist_ok=True)\n",
            "report_path.write_text(json.dumps(VALIDATED_R003_REPORT), encoding='utf-8')\n",
            "CONFIG.update(train_file=INPUTS['train'], test_file=INPUTS['test'], sample_file=INPUTS['sample'])\n",
            "config_path = RUN_ROOT / 'config_r003_final.json'\n",
            "config_path.write_text(json.dumps(CONFIG), encoding='utf-8')\n",
            "subprocess.run([sys.executable, '-m', 'src.final_r003_submission', '--root', str(RUN_ROOT), '--config', str(config_path), '--tag', 'kaggle-replay'], cwd=RUN_ROOT, check=True)\n",
            "print('Inspect RUN_ROOT/submissions, commit this Kaggle notebook version, and upload only with explicit human approval.')\n",
        ]},
    ]
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "cells": cells}
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"
    save_json(output, notebook, exclusive=True)


def build_final_r003_submission(root=ROOT, config_path=None, *, tag="first-live"):
    root = Path(root)
    r003 = read_json(root / "outputs/reports/R003.json")
    if r003["status"] != "completed":
        raise ValueError("R003 must be a completed immutable validation record")
    config = load_config(config_path or root / "config_r003.json")
    if config.to_dict() != r003["config"]:
        raise ValueError("Final refit config differs from frozen R003 configuration")
    if not tag.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Tag must be descriptive letters, digits, underscores, or hyphens")
    package = root / "outputs/final" / f"R003_final_refit_{tag}_{_stamp()}"
    package.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot_source(package / "source")
    save_json(package / "config.json", config.to_dict(), exclusive=True)
    save_json(package / "environment.json", environment(used_packages=[config.model]), exclusive=True)
    save_json(package / "validated_R003_report.json", r003, exclusive=True)

    total_start = time.perf_counter()
    train, train_fingerprint = load_training(config, root)
    if len(train) != 731_942:
        raise ValueError(f"Unexpected final-fit train row count: {len(train)}")
    if train[config.target].isna().any():
        raise ValueError("Final fit contains missing labels")
    model = build_pipeline(config)
    y = train[config.target].map({label: index for index, label in enumerate(config.class_order)})
    fit_start = time.perf_counter()
    model.fit(train[config.features], y)
    fit_seconds = time.perf_counter() - fit_start
    final_model_path = package / "model.pkl"
    with final_model_path.open("xb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    test_path = config.path(root, config.test_file)
    raw_test = read_table(test_path, id_columns=config.id_columns)
    test_fingerprint = {"sha256": sha256(test_path), "rows": len(raw_test), "columns": list(raw_test.columns)}
    if len(raw_test) != 262_648:
        raise ValueError(f"Unexpected test row count: {len(raw_test)}")
    test, causal_checks = _causal_test_features(train, raw_test, config)
    predictions = np.asarray(model_predictions(model, test[config.features], config), dtype=float)
    if predictions.shape != (len(test), 2) or not np.isfinite(predictions).all() or (predictions < 0).any() or (predictions > 1).any():
        raise ValueError("Final probabilities are not finite binary probabilities in [0, 1]")
    prediction_path = package / "test_predictions.csv"
    prediction_metadata = save_predictions(prediction_path, test, predictions, config, test_fingerprint)
    sample_path = config.path(root, config.sample_file)
    sample = read_table(sample_path, id_columns=config.id_columns)
    submission = build_submission(sample, pd.read_csv(prediction_path, dtype={"transaction_id": "string"}), predictions,
                                  prediction_metadata, config)
    if len(submission) != 262_648 or list(submission.columns) != ["transaction_id", "fraud"]:
        raise ValueError("Final submission schema or row count differs from the official contract")
    if not submission["transaction_id"].astype("string").equals(sample["transaction_id"].astype("string")):
        raise ValueError("Submission did not retain exact sample transaction-ID order")
    submission_path = root / "submissions" / f"sub_R003_catboost_final_refit_causal_history_{tag}_{package.name.rsplit('_', 1)[-1]}.csv"
    save_frame(submission_path, submission)
    positive = predictions[:, config.class_order.index(config.positive_class)]
    report = {
        "kind": "R003_final_refit_submission", "validated_experiment": "R003", "validated_r003_report_sha256": sha256(root / "outputs/reports/R003.json"),
        "config": config.to_dict(), "feature_manifest": list(config.features), "source_hashes": source_hashes,
        "environment": read_json(package / "environment.json"), "train_fingerprint": train_fingerprint, "test_fingerprint": test_fingerprint,
        "final_fit": {"rows": len(train), "uses_all_labeled_rows": True, "uses_fold_models": False,
                      "model_path": str(final_model_path.relative_to(root)), "model_sha256": sha256(final_model_path), "runtime_seconds": fit_seconds,
                      "fitted_parameters": model.named_steps["model"].get_params()},
        "causal_test_history": causal_checks,
        "prediction": {"path": str(prediction_path.relative_to(root)), "sha256": prediction_metadata["sha256"], "identity_hash": prediction_metadata["identity_hash"],
                       "rows": len(predictions), "min": float(positive.min()), "max": float(positive.max()), "mean": float(positive.mean()),
                       "quantiles": {str(q): float(np.quantile(positive, q)) for q in (0, .001, .01, .05, .5, .95, .99, .999, 1)}},
        "submission": {"path": str(submission_path.relative_to(root)), "sha256": sha256(submission_path), "rows": len(submission),
                       "columns": list(submission.columns), "sample_sha256": sha256(sample_path), "uploaded": False},
        "runtime_seconds": {"total": time.perf_counter() - total_start, "final_fit": fit_seconds,
                            "test_feature_generation": causal_checks["test_history_generation_seconds"]},
        "causal_assertions": {"no_target_in_test": True, "no_test_rows_in_fit": True, "test_preprocessing_not_fitted": True,
                              "strictly_before_timestamp": True, "tied_timestamps_isolated": True},
    }
    report_path = package / "final_submission_report.json"
    save_json(report_path, report, exclusive=True)
    notebook_path = root / "notebooks" / f"reproduce_R003_final_refit_{tag}_{package.name.rsplit('_', 1)[-1]}.ipynb"
    _export_notebook(package, config, r003, notebook_path)
    report["notebook"] = {"path": str(notebook_path.relative_to(root)), "sha256": sha256(notebook_path),
                          "status": "ready for a human to run and commit as a Kaggle notebook version; not uploaded"}
    save_json(report_path, report)
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tag", default="first-live")
    args = parser.parse_args()
    try:
        print(build_final_r003_submission(args.root, args.config, tag=args.tag))
    except (ValueError, OSError, ImportError, TypeError) as exc:
        parser.exit(2, f"Final R003 submission stopped: {exc}\n")


if __name__ == "__main__":
    main()
