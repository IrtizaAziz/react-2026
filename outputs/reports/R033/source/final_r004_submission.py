"""Immutable all-train R004 refit with causal customer and pair test history."""
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

from .config import ROOT, load_config
from .customer_history import customer_history_from_prior_stream
from .customer_relationships import customer_relationship_from_prior_stream
from .data import add_feature_profile, load_training, read_table, require_columns, validate_feature_profile
from .features import build_pipeline
from .predict import model_predictions, save_predictions
from .submission import build_submission
from .utils import environment, read_json, save_frame, save_json, sha256, snapshot_source


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _causal_test_features(raw_train, raw_test, config):
    require_columns(raw_train, ["timestamp", "customer_id", "device_id", "location", "amount_bdt"], "R004 train state")
    require_columns(raw_test, ["timestamp", "customer_id", "device_id", "location", "amount_bdt", *config.id_columns], "R004 test state")
    if config.target in raw_test:
        raise ValueError("Test input contains a target column")
    train_times, test_times = pd.to_datetime(raw_train["timestamp"], errors="raise"), pd.to_datetime(raw_test["timestamp"], errors="raise")
    if train_times.max() >= test_times.min():
        raise ValueError("Test stream must start strictly after labeled train history")
    order = np.argsort(test_times.to_numpy(dtype="datetime64[ns]"), kind="stable")
    chronological = raw_test.iloc[order]
    start = time.perf_counter()
    customer = customer_history_from_prior_stream(raw_train["timestamp"], raw_train["customer_id"], raw_train["amount_bdt"],
                                                   chronological["timestamp"], chronological["customer_id"], chronological["amount_bdt"])
    relationships = customer_relationship_from_prior_stream(
        raw_train["timestamp"], raw_train["customer_id"], raw_train["device_id"], raw_train["location"],
        chronological["timestamp"], chronological["customer_id"], chronological["device_id"], chronological["location"],
    )
    customer.index, relationships.index = chronological.index, chronological.index
    result = add_feature_profile(raw_test, "react2026_static")
    result = pd.concat([result, customer.reindex(raw_test.index), relationships.reindex(raw_test.index)], axis=1)
    validate_feature_profile(result, config)
    if len(result) != len(raw_test) or not result.index.equals(raw_test.index):
        raise ValueError("Causal test features lost original test order")
    return result, {"train_before_test": True, "test_processed_chronologically": True,
                    "test_timestamp_tie_rows": int(test_times.duplicated(keep=False).sum()),
                    "test_timestamp_batches": int(test_times.nunique()),
                    "test_feature_generation_seconds": time.perf_counter() - start,
                    "raw_state_fields": ["timestamp", "customer_id", "device_id", "location", "amount_bdt"],
                    "same_timestamp_contract": "query/predict full batch before raw-state update", "original_order_restored": True}


def _export_notebook(package, config, r004, output):
    source = {path.name: base64.b64encode(path.read_bytes()).decode("ascii") for path in sorted((package / "source").glob("*.py"))}
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# REACT 2026 — R004 final causal refit\n", "Map only organizer files, run, inspect the CSV, and commit a Kaggle version. This notebook never uploads.\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "import base64, json, subprocess, sys\nfrom pathlib import Path\n",
            "RUN_ROOT = Path('/kaggle/working/react-r004-final').resolve()\n",
            "INPUTS = {'train': '/kaggle/input/REPLACE/train.csv', 'test': '/kaggle/input/REPLACE/test.csv', 'sample': '/kaggle/input/REPLACE/sample_submission.csv'}\n",
            "SOURCES = " + repr(source) + "\nCONFIG = " + repr(config.to_dict()) + "\nVALIDATED_R004_REPORT = " + repr(r004) + "\n",
            "if any('REPLACE' in path for path in INPUTS.values()): raise ValueError('Set organizer input paths first')\n",
            "source_dir = RUN_ROOT / 'src'; source_dir.mkdir(parents=True, exist_ok=False)\n",
            "for name, encoded in SOURCES.items(): (source_dir / name).write_bytes(base64.b64decode(encoded))\n",
            "report_path = RUN_ROOT / 'outputs' / 'reports' / 'R004.json'; report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(json.dumps(VALIDATED_R004_REPORT), encoding='utf-8')\n",
            "CONFIG.update(train_file=INPUTS['train'], test_file=INPUTS['test'], sample_file=INPUTS['sample'])\n",
            "config_path = RUN_ROOT / 'config_r004_final.json'; config_path.write_text(json.dumps(CONFIG), encoding='utf-8')\n",
            "subprocess.run([sys.executable, '-m', 'src.final_r004_submission', '--root', str(RUN_ROOT), '--config', str(config_path), '--tag', 'kaggle-replay'], cwd=RUN_ROOT, check=True)\n",
        ]},
    ]
    for index, cell in enumerate(cells): cell["id"] = f"cell-{index:03d}"
    save_json(output, {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}, exclusive=True)


def build_final_r004_submission(root=ROOT, config_path=None, *, tag="second-live"):
    root = Path(root)
    r004 = read_json(root / "outputs/reports/R004.json")
    config = load_config(config_path or root / "config_r004.json")
    if r004["status"] != "completed" or config.to_dict() != r004["config"]:
        raise ValueError("R004 report/config is not the frozen completed experiment")
    package = root / "outputs/final" / f"R004_final_refit_{tag}_{_stamp()}"
    package.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot_source(package / "source")
    save_json(package / "config.json", config.to_dict(), exclusive=True)
    save_json(package / "environment.json", environment(used_packages=[config.model]), exclusive=True)
    save_json(package / "validated_R004_report.json", r004, exclusive=True)
    start = time.perf_counter()
    train, train_fingerprint = load_training(config, root)
    if len(train) != 731_942 or train[config.target].isna().any(): raise ValueError("Final R004 train input is invalid")
    model, y = build_pipeline(config), train[config.target].map({label: index for index, label in enumerate(config.class_order)})
    fit_start = time.perf_counter(); model.fit(train[config.features], y); fit_seconds = time.perf_counter() - fit_start
    model_path = package / "model.pkl"
    with model_path.open("xb") as handle: pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del train
    train_path, test_path = config.path(root, config.train_file), config.path(root, config.test_file)
    raw_train, raw_test = read_table(train_path, id_columns=config.id_columns), read_table(test_path, id_columns=config.id_columns)
    test_fingerprint = {"sha256": sha256(test_path), "rows": len(raw_test), "columns": list(raw_test.columns)}
    if len(raw_test) != 262_648: raise ValueError("Final R004 test input row count is invalid")
    test, causal = _causal_test_features(raw_train, raw_test, config)
    predictions = np.asarray(model_predictions(model, test[config.features], config), dtype=float)
    if predictions.shape != (len(test), 2) or not np.isfinite(predictions).all() or (predictions < 0).any() or (predictions > 1).any(): raise ValueError("Invalid final probability matrix")
    prediction_path = package / "test_predictions.csv"; metadata = save_predictions(prediction_path, test, predictions, config, test_fingerprint)
    sample_path, sample = config.path(root, config.sample_file), read_table(config.path(root, config.sample_file), id_columns=config.id_columns)
    artifact = pd.read_csv(prediction_path, dtype={"transaction_id": "string"})
    submission = build_submission(sample, artifact, predictions, metadata, config)
    if len(submission) != 262_648 or list(submission.columns) != ["transaction_id", "fraud"] or not submission["transaction_id"].astype("string").equals(sample["transaction_id"].astype("string")): raise ValueError("Final submission schema/alignment failed")
    submission_path = root / "submissions" / f"sub_R004_catboost_final_refit_causal_relationships_{tag}_{package.name.rsplit('_', 1)[-1]}.csv"; save_frame(submission_path, submission)
    positive = predictions[:, config.class_order.index(config.positive_class)]
    report = {"kind": "R004_final_refit_submission", "validated_experiment": "R004", "validated_R004_report_sha256": sha256(root / "outputs/reports/R004.json"),
              "config": config.to_dict(), "feature_manifest": list(config.features), "source_hashes": source_hashes, "environment": read_json(package / "environment.json"), "train_fingerprint": train_fingerprint, "test_fingerprint": test_fingerprint,
              "final_fit": {"rows": 731942, "uses_all_labeled_rows": True, "uses_fold_models": False, "model_path": str(model_path.relative_to(root)), "model_sha256": sha256(model_path), "runtime_seconds": fit_seconds, "fitted_parameters": model.named_steps["model"].get_params()},
              "causal_test_history": causal, "prediction": {"path": str(prediction_path.relative_to(root)), "sha256": metadata["sha256"], "identity_hash": metadata["identity_hash"], "rows": len(predictions), "min": float(positive.min()), "max": float(positive.max()), "mean": float(positive.mean()), "quantiles": {str(q): float(np.quantile(positive, q)) for q in (0, .001, .01, .05, .5, .95, .99, .999, 1)}},
              "submission": {"path": str(submission_path.relative_to(root)), "sha256": sha256(submission_path), "rows": len(submission), "columns": list(submission.columns), "sample_sha256": sha256(sample_path), "uploaded": False},
              "runtime_seconds": {"total": time.perf_counter() - start, "final_fit": fit_seconds, "test_feature_generation": causal["test_feature_generation_seconds"]},
              "causal_assertions": {"no_target_in_test": True, "no_test_rows_in_fit": True, "test_preprocessing_not_fitted": True, "strictly_before_timestamp": True, "tied_timestamps_isolated": True}}
    report_path = package / "final_submission_report.json"; save_json(report_path, report, exclusive=True)
    notebook = root / "notebooks" / f"reproduce_R004_final_refit_{tag}_{package.name.rsplit('_', 1)[-1]}.ipynb"; _export_notebook(package, config, r004, notebook)
    report["notebook"] = {"path": str(notebook.relative_to(root)), "sha256": sha256(notebook), "status": "ready for human Kaggle execution/version commit; not uploaded"}; save_json(report_path, report)
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=ROOT); parser.add_argument("--config", type=Path); parser.add_argument("--tag", default="second-live")
    args = parser.parse_args()
    try: print(build_final_r004_submission(args.root, args.config, tag=args.tag))
    except (ValueError, OSError, ImportError, TypeError) as exc: parser.exit(2, f"Final R004 submission stopped: {exc}\n")


if __name__ == "__main__": main()
