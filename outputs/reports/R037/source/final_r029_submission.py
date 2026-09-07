"""Dedicated immutable R029 all-train refit and local submission export."""
from __future__ import annotations

import hashlib, importlib, importlib.util, json, pickle, shutil, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MATRIX_SHA = "ab767d22d77a44ef3514e8580d9287be720b210475037158514883c1c249771c"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def load_frozen_source(root: Path):
    source = root / "outputs/reports/R029/source"
    package_name = f"_r029_final_{stamp().lower()}"
    temp = Path(tempfile.mkdtemp(prefix="r029_final_source_"))
    package_dir = temp / package_name
    shutil.copytree(source, package_dir)
    spec = importlib.util.spec_from_file_location(package_name, package_dir / "__init__.py", submodule_search_locations=[str(package_dir)])
    module = importlib.util.module_from_spec(spec); sys.modules[package_name] = module; spec.loader.exec_module(module)
    return temp, package_name


def source_hashes(directory: Path):
    return {p.name: sha256(p) for p in sorted(directory.glob("*.py"))}


def build_test_features(raw_train, raw_test, config, pkg):
    data = importlib.import_module(f"{pkg}.data")
    history_mod = importlib.import_module(f"{pkg}.customer_history")
    rel_mod = importlib.import_module(f"{pkg}.customer_relationships")
    velocity_mod = importlib.import_module(f"{pkg}.velocity")
    cm_mod = importlib.import_module(f"{pkg}.customer_merchant")
    mh_mod = importlib.import_module(f"{pkg}.merchant_history")
    new_mod = importlib.import_module(f"{pkg}.merchant_new_customers")

    train_time = pd.to_datetime(raw_train.timestamp, errors="raise")
    test_time = pd.to_datetime(raw_test.timestamp, errors="raise")
    if train_time.max() >= test_time.min():
        raise ValueError("Train does not end strictly before test")
    order = np.argsort(test_time.to_numpy(dtype="datetime64[ns]"), kind="stable")
    chronological = raw_test.iloc[order].reset_index(drop=False).rename(columns={"index": "__original_index__"})
    combined = pd.concat([raw_train.drop(columns=[config["target"]]), chronological.drop(columns=["__original_index__"])], ignore_index=True)

    history = history_mod.customer_history_production(combined.timestamp, combined.customer_id, combined.amount_bdt)
    relationships = rel_mod.customer_relationship_production(combined.timestamp, combined.customer_id, combined.device_id, combined.location)
    devices = rel_mod.device_global_production(combined.timestamp, combined.customer_id, combined.device_id)
    velocity = velocity_mod.velocity_production(combined.timestamp, combined.customer_id, combined.device_id)
    familiarity = cm_mod.customer_merchant_production(combined.timestamp, combined.customer_id, combined.merchant_id, history.customer_prior_count)
    merchant = mh_mod.merchant_history_production(combined.timestamp, combined.customer_id, combined.merchant_id)
    new = new_mod.build_features(combined, ["merchant_new_customer_share_24h", "merchant_seconds_since_last_new_customer"])

    cut = len(raw_train)
    state_frames = [x.iloc[cut:].reset_index(drop=True) for x in [history, relationships, devices, velocity, familiarity, merchant]]
    static = data.add_feature_profile(chronological.drop(columns=["__original_index__"]), "react2026_static").reset_index(drop=True)
    new_test = new.iloc[cut:].reset_index(drop=True)
    result = pd.concat([static, *state_frames, new_test], axis=1)
    result.index = chronological["__original_index__"].to_numpy()
    result = result.reindex(raw_test.index)
    if list(result[config["features"]].columns) != list(config["features"]):
        raise ValueError("R029 test feature order mismatch")
    return result, {"timestamp_batches": int(test_time.nunique()), "tie_rows": int(test_time.duplicated(keep=False).sum())}


def main():
    started = time.perf_counter()
    root = ROOT
    report_dir = root / "outputs/reports/R029"
    config = json.loads((report_dir / "config.json").read_text(encoding="utf-8"))
    matrix_path = root / "outputs/feature_matrices/R029.pkl"
    matrix_manifest = json.loads((root / "outputs/feature_matrices/R029.json").read_text(encoding="utf-8"))
    if sha256(matrix_path) != EXPECTED_MATRIX_SHA or matrix_manifest.get("matrix_sha256") != EXPECTED_MATRIX_SHA:
        raise ValueError("R029 saved matrix hash mismatch")
    if len(config["features"]) != 52 or config["features"][-2:] != ["merchant_new_customer_share_24h", "merchant_seconds_since_last_new_customer"]:
        raise ValueError("R029 exact 52-feature manifest mismatch")
    immutable_params = config["model_params"]
    if config["model"] != "catboost" or config["predict_test"] or config["aggregation"] is not None:
        raise ValueError("Immutable R029 config is not an eligible final refit recipe")

    temp, pkg = load_frozen_source(root)
    try:
        frozen_source = root / "outputs/reports/R029/source"
        raw_train = pd.read_csv(root / config["train_file"], dtype={c: "string" for c in config["id_columns"]})
        raw_test = pd.read_csv(root / config["test_file"], dtype={c: "string" for c in config["id_columns"]})
        sample = pd.read_csv(root / config["sample_file"], dtype={"transaction_id": "string"})
        if "fraud" in raw_test or list(sample.columns) != ["transaction_id", "fraud"]:
            raise ValueError("Forbidden test target or invalid sample schema")
        if len(sample) != len(raw_test) or not sample.transaction_id.equals(raw_test.transaction_id):
            raise ValueError("Sample/test alignment fails before fitting")

        saved = pd.read_pickle(matrix_path)
        parent = pd.read_pickle(root / "outputs/feature_matrices/R026.pkl")
        new_mod = importlib.import_module(f"{pkg}.merchant_new_customers")
        new_value = new_mod.build_features(raw_train, ["merchant_seconds_since_last_new_customer"])
        reconstructed = parent.copy()
        reconstructed["merchant_seconds_since_last_new_customer"] = new_value.iloc[:, 0].to_numpy(copy=False)
        # Compare the complete immutable frame, then the exact configured model view.
        pd.testing.assert_index_equal(reconstructed.index, saved.index, exact=True)
        if not reconstructed["transaction_id"].astype("string").equals(saved["transaction_id"].astype("string")):
            raise ValueError("R029 transaction ID/order parity failed")
        pd.testing.assert_frame_equal(reconstructed, saved, check_dtype=True, check_exact=True)
        train_features = reconstructed[config["features"]].copy()
        if list(train_features.columns) != list(config["features"]):
            raise ValueError("R029 train feature order differs")
        if matrix_manifest["train_fingerprint"] != {"sha256": sha256(root / config["train_file"]), "rows": len(raw_train), "columns": list(raw_train.columns)}:
            raise ValueError("R029 train fingerprint differs")
        parity = {"passed": True, "saved_matrix_sha256": EXPECTED_MATRIX_SHA, "rows": len(saved), "features": len(config["features"]), "ids_order": True, "feature_order": True, "values_dtypes_exact": True, "train_fingerprint": matrix_manifest["train_fingerprint"]}

        test_features, causal = build_test_features(raw_train, raw_test, config, pkg)
        if len(test_features) != len(raw_test) or not test_features.index.equals(raw_test.index):
            raise ValueError("Test feature row/order restoration failed")

        package = root / "outputs/final" / f"R029_final_refit_{stamp()}"
        package.mkdir(parents=True, exist_ok=False)
        source_dst = package / "source"; shutil.copytree(frozen_source, source_dst)
        launcher_dst = package / "final_r029_submission.py"; shutil.copy2(Path(__file__), launcher_dst)
        (package / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (package / "feature_manifest.json").write_text(json.dumps({"features": config["features"], "categorical_features": config["categorical_features"], "count": 52}, indent=2) + "\n", encoding="utf-8")
        (package / "matrix_parity.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
        (package / "environment.json").write_text(json.dumps({"python": sys.version, "platform": sys.platform, "packages": matrix_manifest.get("frozen_source_provenance", {}), "matrix_sha256": EXPECTED_MATRIX_SHA}, indent=2) + "\n", encoding="utf-8")

        from sklearn.pipeline import Pipeline
        from .config import Config
        frozen_config = Config(**config)
        features_mod = importlib.import_module(f"{pkg}.features")
        model = features_mod.build_pipeline(frozen_config)
        actual_params = dict(immutable_params); actual_params.pop("cat_features", None)
        if model.named_steps["model"].get_params()["iterations"] != actual_params["iterations"] or model.named_steps["model"].get_params()["depth"] != actual_params["depth"] or model.named_steps["model"].get_params()["learning_rate"] != actual_params["learning_rate"] or model.named_steps["model"].get_params()["l2_leaf_reg"] != actual_params["l2_leaf_reg"]:
            raise ValueError("CatBoost immutable parameters differ")
        y = raw_train[config["target"]].astype(int).to_numpy()
        model.fit(train_features, y)
        model_path = package / "R029_final_fit_model.pkl"
        with model_path.open("xb") as f: pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        pred = np.asarray(model.predict_proba(test_features[config["features"]]), dtype=float)
        if pred.shape != (len(raw_test), 2) or not np.isfinite(pred).all() or (pred < 0).any() or (pred > 1).any():
            raise ValueError("Predictions are not finite probabilities in [0,1]")
        pred_path = package / "test_predictions.csv"
        pd.DataFrame({"transaction_id": raw_test.transaction_id, "pred_0": pred[:, 0], "pred_1": pred[:, 1]}).to_csv(pred_path, index=False)
        submission_path = root / "submissions" / f"submission_R029_{stamp()}.csv"
        submission = pd.DataFrame({"transaction_id": sample.transaction_id, "fraud": pred[:, 1]})
        submission.to_csv(submission_path, index=False)
        if list(submission.columns) != ["transaction_id", "fraud"] or len(submission) != len(sample) or not submission.transaction_id.equals(sample.transaction_id) or submission.fraud.isna().any() or not np.isfinite(submission.fraud).all() or ((submission.fraud < 0) | (submission.fraud > 1)).any() or submission.transaction_id.duplicated().any():
            raise ValueError("Final submission validation failed")
        test_fp = {"sha256": sha256(root / config["test_file"]), "rows": len(raw_test), "columns": list(raw_test.columns)}
        metadata = {"prediction_kind": "probability", "columns": ["transaction_id", "pred_0", "pred_1"], "rows": len(pred), "test_fingerprint": test_fp, "test_fitting": False, "causal": causal, "sha256": sha256(pred_path)}
        (package / "prediction_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (package / "train_test_fingerprints.json").write_text(json.dumps({"train": matrix_manifest["train_fingerprint"], "test": test_fp, "sample": {"sha256": sha256(root / config["sample_file"]), "rows": len(sample)}}, indent=2) + "\n", encoding="utf-8")
        report = {"kind": "R029_final_refit_submission", "immutable_r029": True, "matrix_parity": parity, "model_parameters": immutable_params, "final_model": {"path": str(model_path.relative_to(root)), "sha256": sha256(model_path), "all_labeled_train": True, "test_fitting": False}, "test_predictions": {"path": str(pred_path.relative_to(root)), "sha256": sha256(pred_path), "rows": len(pred), "min": float(pred[:,1].min()), "max": float(pred[:,1].max())}, "submission": {"path": str(submission_path.relative_to(root)), "sha256": sha256(submission_path), "schema": list(submission.columns), "rows": len(submission), "sample_order_exact": True, "uploaded": False}, "causal": {"train_seeded_raw_state": True, "test_processed_chronologically": True, "strict_less_than_timestamp": True, "equal_timestamp_isolation": True, "test_labels_used": False, "test_fitted_statistics": False, **causal}, "source_hashes": source_hashes(source_dst), "launcher_sha256": sha256(launcher_dst), "elapsed_seconds": time.perf_counter() - started}
        (package / "final_submission_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        for name in [n for n in list(sys.modules) if n == pkg or n.startswith(pkg + ".")]: sys.modules.pop(name, None)
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    main()
