"""Immutable R013-only final-fit, train-seeded causal submission export."""
import hashlib
import json
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE))

import numpy as np
import pandas as pd
from src.config import load_config
from src.customer_history import customer_history_from_prior_stream, customer_history_production
from src.customer_merchant import customer_merchant_production
from src.customer_relationships import customer_relationship_from_prior_stream, device_global_from_prior_stream
from src.data import add_feature_profile, load_training, read_table, require_columns, validate_feature_profile
from src.features import build_pipeline
from src.predict import model_predictions, save_predictions
from src.submission import build_submission
from src.utils import git_info, save_json, sha256
from src.velocity import velocity_from_prior_stream


def source_hashes(directory):
    return {p.name: sha256(p) for p in sorted(directory.glob("*.py"))}


def main():
    record = json.loads((PACKAGE / "validated_R013_report.json").read_text())
    config = load_config(PACKAGE / "config.json")
    config.validate()
    if record["experiment_id"] != "R013" or record["status"] != "completed":
        raise ValueError("R013 completed record is required")
    if config.to_dict() != record["config"]:
        raise ValueError("Frozen R013 config does not match its record")
    hashes = source_hashes(PACKAGE / "src")
    if hashes != record["source_hashes"]:
        raise ValueError("Frozen R013 source hashes do not match its record")
    foundation = json.loads((ROOT / "reports" / "live_foundation.json").read_text())
    train_path, test_path, sample_path = (config.path(ROOT, config.train_file), config.path(ROOT, config.test_file), config.path(ROOT, config.sample_file))
    if sha256(train_path) != foundation["input_hashes"]["train.csv"] or sha256(test_path) != foundation["input_hashes"]["test.csv"]:
        raise ValueError("Competition train/test hashes differ from the foundation")
    if sha256(train_path) != record["train_fingerprint"]["sha256"]:
        raise ValueError("Training hash differs from immutable R013")
    start = time.perf_counter()
    train, train_fp = load_training(config, ROOT)
    if train_fp != record["train_fingerprint"]:
        raise ValueError("R013 training fingerprint mismatch")
    model = build_pipeline(config)
    y = train[config.target].map({label: index for index, label in enumerate(config.class_order)})
    fit_start = time.perf_counter(); model.fit(train[config.features], y); fit_seconds = time.perf_counter() - fit_start
    model_path = PACKAGE / "R013_final_fit_model.pkl"
    with model_path.open("xb") as handle: pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    raw_train = read_table(train_path, id_columns=config.id_columns)
    raw_test = read_table(test_path, id_columns=config.id_columns)
    if config.target in raw_test: raise ValueError("Test must not contain a target")
    train_time, test_time = pd.to_datetime(raw_train.timestamp, errors="raise"), pd.to_datetime(raw_test.timestamp, errors="raise")
    if train_time.max() >= test_time.min(): raise ValueError("Test must be strictly after train")
    order = np.argsort(test_time.to_numpy(dtype="datetime64[ns]"), kind="stable")
    chronological = raw_test.iloc[order]
    customer = customer_history_from_prior_stream(raw_train.timestamp, raw_train.customer_id, raw_train.amount_bdt, chronological.timestamp, chronological.customer_id, chronological.amount_bdt)
    relationships = customer_relationship_from_prior_stream(raw_train.timestamp, raw_train.customer_id, raw_train.device_id, raw_train.location, chronological.timestamp, chronological.customer_id, chronological.device_id, chronological.location)
    devices = device_global_from_prior_stream(raw_train.timestamp, raw_train.customer_id, raw_train.device_id, chronological.timestamp, chronological.customer_id, chronological.device_id)
    velocity = velocity_from_prior_stream(raw_train.timestamp, raw_train.customer_id, raw_train.device_id, chronological.timestamp, chronological.customer_id, chronological.device_id)
    train_history = customer_history_production(raw_train.timestamp, raw_train.customer_id, raw_train.amount_bdt)
    merchant = customer_merchant_production(pd.concat([raw_train.timestamp, chronological.timestamp], ignore_index=True), pd.concat([raw_train.customer_id, chronological.customer_id], ignore_index=True), pd.concat([raw_train.merchant_id, chronological.merchant_id], ignore_index=True), pd.concat([train_history.customer_prior_count, customer.customer_prior_count], ignore_index=True)).iloc[len(raw_train):].reset_index(drop=True)
    for frame in (customer, relationships, devices, velocity, merchant): frame.index = chronological.index
    test = add_feature_profile(raw_test, "react2026_static")
    test = pd.concat([test, customer.reindex(raw_test.index), relationships.reindex(raw_test.index), devices.reindex(raw_test.index), velocity.reindex(raw_test.index), merchant.reindex(raw_test.index)], axis=1)
    validate_feature_profile(test, config)
    if list(test[config.features].columns) != list(config.features) or len(config.features) != 45: raise ValueError("R013 feature manifest/order mismatch")
    predictions = np.asarray(model_predictions(model, test[config.features], config), dtype=float)
    if predictions.shape != (len(raw_test), 2) or not np.isfinite(predictions).all() or (predictions < 0).any() or (predictions > 1).any(): raise ValueError("Invalid probability output")
    test_fp = {"sha256": sha256(test_path), "rows": len(raw_test), "columns": list(raw_test.columns)}
    prediction_path = PACKAGE / "test_predictions.csv"; metadata = save_predictions(prediction_path, test, predictions, config, test_fp)
    sample = read_table(sample_path, id_columns=config.id_columns)
    submission = build_submission(sample, pd.read_csv(prediction_path, dtype={"transaction_id": "string"}), predictions, metadata, config)
    if list(submission.columns) != ["transaction_id", "fraud"] or len(submission) != len(raw_test) or not submission.transaction_id.astype("string").equals(sample.transaction_id.astype("string")): raise ValueError("Submission schema/alignment mismatch")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    submission_path = ROOT / "submissions" / f"submission_R013_{stamp}.csv"
    if submission_path.exists(): raise FileExistsError(submission_path)
    submission_path.parent.mkdir(exist_ok=True); submission.to_csv(submission_path, index=False)
    positive = predictions[:, config.class_order.index(config.positive_class)]
    report = {"selected_experiment_id": "R013", "profile": config.feature_profile, "config_hash": sha256(PACKAGE / "config.json"), "source_hashes": hashes, "train_hash": train_fp["sha256"], "test_hash": test_fp["sha256"], "feature_manifest_hash": hashlib.sha256(json.dumps(config.features, separators=(",", ":")).encode()).hexdigest(), "model_final_fit_seed": config.seed, "prediction": {"min": float(positive.min()), "max": float(positive.max()), "mean": float(positive.mean()), "rows": len(positive)}, "submission": {"path": str(submission_path), "sha256": sha256(submission_path), "columns": list(submission.columns), "sample_sha256": sha256(sample_path)}, "checks": {"foundation_train_test_hashes": True, "r013_config_source_immutable": True, "r013_exact_manifest": True, "train_seeded_sequential_test_history": True, "same_timestamp_test_rows_isolated": True, "test_feature_order_matches_training": True, "prediction_order_is_test_transaction_order": True, "finite_bounded_predictions": True, "sample_id_alignment": True, "two_column_schema": True, "prior_submission_not_overwritten": True}, "final_fit": {"uses_all_labeled_train_rows": True, "uses_fold_models": False, "model_path": str(model_path), "model_sha256": sha256(model_path), "seconds": fit_seconds}, "git": git_info(ROOT), "total_seconds": time.perf_counter() - start}
    save_json(PACKAGE / "submission_R013_provenance.json", report, exclusive=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
