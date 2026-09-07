"""Immutable all-train R017 refit and train-seeded causal Kaggle export; never uploads."""
import argparse
import hashlib
import json
import pickle
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    hash_ = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): hash_.update(block)
    return hash_.hexdigest()


def stamp(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def source_hashes(directory): return {path.name: digest(path) for path in sorted(Path(directory).glob("*.py"))}


def build_final(root):
    root = Path(root); record_path = root / "outputs/reports/R017.json"; record = json.loads(record_path.read_text())
    if record["experiment_id"] != "R017" or record["status"] != "completed": raise ValueError("Completed immutable R017 is required")
    frozen_source = root / "outputs/reports/R017/source"
    if source_hashes(frozen_source) != record["source_hashes"]: raise ValueError("R017 frozen source hashes disagree with immutable record")
    frozen_config = root / "outputs/reports/R017/config.json"
    package = root / "outputs/final" / f"R017_final_refit_{stamp()}"; package.mkdir(parents=True, exist_ok=False)
    shutil.copytree(frozen_source, package / "src")
    shutil.copy2(frozen_config, package / "config.json"); shutil.copy2(record_path, package / "validated_R017_report.json")
    shutil.copy2(root / "outputs/reports/R017/decision_report.json", package / "post_fit_corrected_diagnostic_report.json")
    if source_hashes(package / "src") != record["source_hashes"]: raise ValueError("Copied R017 source hash verification failed")
    sys.path.insert(0, str(package))
    # This launcher is outside the frozen package; discard its already-imported
    # workspace package so every model/feature import below resolves to R017's snapshot.
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."): del sys.modules[name]
    import numpy as np
    import pandas as pd
    from src.config import load_config
    from src.data import add_feature_profile, read_table, validate_feature_profile
    from src.features import build_pipeline
    from src.predict import model_predictions, save_predictions
    from src.submission import build_submission
    from src.customer_history import customer_history_production
    from src.customer_relationships import customer_relationship_production, device_global_production
    from src.velocity import velocity_production
    from src.customer_merchant import customer_merchant_production
    from src.merchant_history import merchant_history_production
    config = load_config(package / "config.json"); config.validate()
    if config.to_dict() != record["config"] or len(config.features) != 50: raise ValueError("R017 config/feature manifest mismatch")
    expected_tail = ["merchant_prior_transaction_count", "merchant_seconds_since_last", "merchant_prior_unique_customers", "merchant_transactions_24h", "merchant_unique_customers_24h"]
    if config.features[-5:] != expected_tail: raise ValueError("R017 merchant feature tail mismatch")
    train_path, test_path, sample_path = (root / config.train_file, root / config.test_file, root / config.sample_file)
    foundation = json.loads((root / "reports/live_foundation.json").read_text())
    if digest(train_path) != record["train_fingerprint"]["sha256"] or digest(train_path) != foundation["input_hashes"]["train.csv"] or digest(test_path) != foundation["input_hashes"]["test.csv"]: raise ValueError("Input hash mismatch")
    start = time.perf_counter(); raw_train = read_table(train_path, id_columns=config.id_columns); raw_test = read_table(test_path, id_columns=config.id_columns)
    if config.target in raw_test: raise ValueError("Test contains forbidden target")
    train_time, test_time = pd.to_datetime(raw_train.timestamp, errors="raise"), pd.to_datetime(raw_test.timestamp, errors="raise")
    if train_time.max() >= test_time.min(): raise ValueError("Train must end strictly before test")
    order = np.argsort(test_time.to_numpy(dtype="datetime64[ns]"), kind="stable"); chronological = raw_test.iloc[order].reset_index(drop=False).rename(columns={"index": "__original_index__"})
    # This continuous raw-only stream seeds state with all train events, then updates only after each full test timestamp batch.
    combined = pd.concat([raw_train.drop(columns=[config.target]), chronological.drop(columns=["__original_index__"])], ignore_index=True)
    history = customer_history_production(combined.timestamp, combined.customer_id, combined.amount_bdt)
    relationships = customer_relationship_production(combined.timestamp, combined.customer_id, combined.device_id, combined.location)
    devices = device_global_production(combined.timestamp, combined.customer_id, combined.device_id)
    velocity = velocity_production(combined.timestamp, combined.customer_id, combined.device_id)
    familiarity = customer_merchant_production(combined.timestamp, combined.customer_id, combined.merchant_id, history.customer_prior_count)
    merchant = merchant_history_production(combined.timestamp, combined.customer_id, combined.merchant_id)
    cut = len(raw_train); state_frames = [history.iloc[cut:].reset_index(drop=True), relationships.iloc[cut:].reset_index(drop=True), devices.iloc[cut:].reset_index(drop=True), velocity.iloc[cut:].reset_index(drop=True), familiarity.iloc[cut:].reset_index(drop=True), merchant.iloc[cut:].reset_index(drop=True)]
    static = add_feature_profile(chronological.drop(columns=["__original_index__"]), "react2026_static").reset_index(drop=True)
    chronological_features = pd.concat([static, *state_frames], axis=1)
    chronological_features.index = chronological["__original_index__"].to_numpy(); test = chronological_features.reindex(raw_test.index)
    validate_feature_profile(test, config)
    if len(test) != len(raw_test) or not test.index.equals(raw_test.index) or list(test[config.features].columns) != list(config.features): raise ValueError("Test row or exact 50-feature order mismatch")
    # Train feature generation uses the same frozen profile; labels enter only the final model fit below.
    train = read_table(train_path, id_columns=config.id_columns)
    train_features = add_feature_profile(train.copy(), config.feature_profile)
    if list(train_features[config.features].columns) != list(config.features): raise ValueError("Final train feature manifest mismatch")
    model = build_pipeline(config); y = train[config.target].map({label: index for index, label in enumerate(config.class_order)})
    fit_start = time.perf_counter(); model.fit(train_features[config.features], y); fit_seconds = time.perf_counter() - fit_start
    model_path = package / "R017_final_fit_model.pkl"
    with model_path.open("xb") as handle: pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    predictions = np.asarray(model_predictions(model, test[config.features], config), dtype=float)
    positive = predictions[:, config.class_order.index(config.positive_class)]
    if predictions.shape != (len(raw_test), 2) or not np.isfinite(predictions).all() or (predictions < 0).any() or (predictions > 1).any(): raise ValueError("Predictions are not finite bounded probabilities")
    test_fp = {"sha256": digest(test_path), "rows": len(raw_test), "columns": list(raw_test.columns)}
    prediction_path = package / "test_predictions.csv"; metadata = save_predictions(prediction_path, test, predictions, config, test_fp)
    sample = read_table(sample_path, id_columns=config.id_columns); artifact = pd.read_csv(prediction_path, dtype={"transaction_id": "string"})
    submission = build_submission(sample, artifact, predictions, metadata, config)
    if list(submission.columns) != ["transaction_id", "fraud"] or len(submission) != len(raw_test) or not submission.transaction_id.astype("string").equals(sample.transaction_id.astype("string")): raise ValueError("Kaggle schema or sample ID alignment mismatch")
    submission_path = root / "submissions" / f"submission_R017_{stamp()}.csv"
    if submission_path.exists(): raise FileExistsError(submission_path)
    submission.to_csv(submission_path, index=False)
    report = {"kind": "R017_final_refit_submission", "validated_experiment": "R017", "validated_R017_report_sha256": digest(record_path), "immutable_r017_unchanged": True, "config": config.to_dict(), "feature_manifest": config.features, "feature_manifest_hash": hashlib.sha256(json.dumps(config.features, separators=(",", ":")).encode()).hexdigest(), "source_hashes": source_hashes(package / "src"), "train_fingerprint": {"sha256": digest(train_path), "rows": len(raw_train), "columns": list(raw_train.columns)}, "test_fingerprint": test_fp, "final_fit": {"uses_all_labeled_train_rows": True, "uses_fold_models": False, "model_path": str(model_path.relative_to(root)), "model_sha256": digest(model_path), "seconds": fit_seconds}, "causal_test_history": {"train_seeded_raw_event_state": True, "train_fraud_not_used_in_state": True, "test_processed_in_timestamp_order": True, "strictly_before_timestamp": True, "same_timestamp_test_rows_isolated": True, "test_timestamp_tie_rows": int(test_time.duplicated(keep=False).sum()), "test_timestamp_batches": int(test_time.nunique()), "original_test_row_order_restored": True}, "prediction": {"path": str(prediction_path.relative_to(root)), "sha256": metadata["sha256"], "rows": len(positive), "min": float(positive.min()), "max": float(positive.max()), "mean": float(positive.mean())}, "submission": {"path": str(submission_path.relative_to(root)), "sha256": digest(submission_path), "rows": len(submission), "columns": list(submission.columns), "sample_sha256": digest(sample_path), "uploaded": False}, "checks": {"r017_source_config_manifest_hashes": True, "train_test_hashes": True, "exact_50_feature_order": True, "causal_train_to_test_continuation": True, "same_timestamp_isolation": True, "row_transaction_alignment": True, "finite_bounded_predictions": True, "exact_sample_submission_id_order": True, "exact_kaggle_schema": True, "post_fit_corrected_diagnostic_preserved": True}, "total_seconds": time.perf_counter() - start}
    (package / "final_submission_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2)); return package / "final_submission_report.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); args = parser.parse_args(); build_final(args.root)
