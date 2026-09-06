"""Readers never discover or load test data as a side effect of training."""
from pathlib import Path
import numpy as np
import pandas as pd
import time
from .utils import sha256

TABLE_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".pq", ".xlsx", ".xls"}


def read_table(path, *, limit=None, id_columns=()):
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv"}:
            return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", nrows=limit,
                               dtype={column: "string" for column in id_columns})
        if suffix == ".jsonl":
            frame = pd.read_json(path, lines=True, nrows=limit)
        elif suffix == ".json":
            frame = pd.read_json(path)
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported table format: {suffix}")
    except ImportError as exc:
        package = "pyarrow" if suffix in {".parquet", ".pq"} else "openpyxl" if suffix == ".xlsx" else "xlrd"
        raise ImportError(f"Reading {suffix} needs an optional reader: pip install {package}") from exc
    for column in id_columns:
        if column in frame:
            frame[column] = frame[column].astype("string")
    return frame if limit is None else frame.head(limit)


def require_columns(frame, columns, context):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{context}: missing columns {missing}")


def add_row_features(frame, row_features):
    """Add only explicitly configured, deterministic per-row features."""
    row_features = set(row_features)
    if not row_features:
        return frame
    if {"FamilySize", "IsAlone"} & row_features:
        require_columns(frame, ["SibSp", "Parch"], "Row-local feature inputs")
    if "Title" in row_features:
        require_columns(frame, ["Name"], "Row-local feature inputs")
    # The caller owns a freshly-read raw table. A shallow copy avoids duplicating
    # its large string columns while new approved columns are appended.
    result = frame.copy(deep=False)
    if {"FamilySize", "IsAlone"} & row_features:
        family_size = result["SibSp"] + result["Parch"] + 1
        if "FamilySize" in row_features:
            result["FamilySize"] = family_size
        if "IsAlone" in row_features:
            result["IsAlone"] = (family_size == 1).astype(int)
    if "Title" in row_features:
        result["Title"] = result["Name"].astype("string").str.extract(r",\s*([^.]*)\.", expand=False)
    return result


REACT2026_STATIC_FEATURES = [
    "amount_bdt", "log_amount_bdt", "account_age_days", "hour", "weekday", "is_weekend",
    "merchant_category", "device_type", "location", "payment_method", "transaction_type",
    "merchant_category_missing", "device_type_missing", "location_missing",
]
REACT2026_STATIC_CATEGORICALS = ["merchant_category", "device_type", "location", "payment_method", "transaction_type"]
REACT2026_CUSTOMER_HISTORY_FEATURES = [
    "customer_prior_count", "customer_first_seen", "customer_seconds_since_last", "customer_observed_age_seconds",
    "customer_prior_mean_amount", "customer_prior_mean_log_amount", "customer_prior_std_log_amount",
    "customer_amount_to_prior_mean", "customer_log_amount_minus_prior_mean", "customer_log_amount_zscore",
]
REACT2026_CUSTOMER_RELATIONSHIP_FEATURES = [
    "customer_device_prior_count", "customer_device_new", "customer_device_seconds_since_last", "customer_device_count_share",
    "customer_location_prior_count", "customer_location_new", "customer_location_seconds_since_last", "customer_location_count_share",
]
REACT2026_DEVICE_GLOBAL_FEATURES = [
    "device_prior_count", "device_prior_distinct_customer_count",
    "device_prior_distinct_customer_count_excluding_current", "device_seconds_since_last",
    "device_observed_age_seconds", "device_prior_other_customer_transaction_count",
    "customer_share_of_device_prior_transactions",
]
REACT2026_VELOCITY_FEATURES = ["customer_prior_1h_count", "device_prior_1h_count"]


def add_feature_profile(frame, profile):
    """Materialize approved deterministic features without observing other rows."""
    if profile is None:
        return frame
    if profile not in {"react2026_static", "react2026_static_customer_history", "react2026_static_customer_history_relationships", "react2026_static_customer_history_relationships_device_global", "react2026_static_customer_history_relationships_device_global_velocity"}:
        raise ValueError(f"Unknown feature profile: {profile}")
    require_columns(frame, ["timestamp", "amount_bdt", "account_age_days", *REACT2026_STATIC_CATEGORICALS], "REACT 2026 static inputs")
    result = frame.copy()
    timestamp = pd.to_datetime(result["timestamp"], errors="raise")
    result["log_amount_bdt"] = np.log1p(result["amount_bdt"].astype(float))
    result["hour"] = timestamp.dt.hour.astype("int8")
    result["weekday"] = timestamp.dt.dayofweek.astype("int8")
    result["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype("int8")
    for column in ("merchant_category", "device_type", "location"):
        result[f"{column}_missing"] = result[column].isna().astype("int8")
    if profile == "react2026_static_customer_history_relationships":
        require_columns(result, ["customer_id", "device_id", "location"], "REACT 2026 customer-relationship inputs")
        from .customer_relationships import customer_relationship_production
        relationship_start = time.perf_counter()
        relationships = customer_relationship_production(result["timestamp"], result["customer_id"], result["device_id"], result["location"])
        for column in relationships:
            result[column] = relationships[column].to_numpy(copy=False)
        result.attrs["customer_relationship_generation_seconds"] = time.perf_counter() - relationship_start
    if profile in {"react2026_static_customer_history_relationships_device_global", "react2026_static_customer_history_relationships_device_global_velocity"}:
        require_columns(result, ["customer_id", "device_id", "location"], "REACT 2026 R005 relationship inputs")
        from .customer_relationships import customer_relationship_production, device_global_production
        relationship_start = time.perf_counter()
        relationships = customer_relationship_production(result["timestamp"], result["customer_id"], result["device_id"], result["location"])
        for column in relationships:
            result[column] = relationships[column].to_numpy(copy=False)
        result.attrs["customer_relationship_generation_seconds"] = time.perf_counter() - relationship_start
        device_start = time.perf_counter()
        devices = device_global_production(result["timestamp"], result["customer_id"], result["device_id"])
        for column in devices:
            result[column] = devices[column].to_numpy(copy=False)
        result.attrs["device_global_generation_seconds"] = time.perf_counter() - device_start
    if profile == "react2026_static_customer_history_relationships_device_global_velocity":
        from .velocity import velocity_production
        velocity_start = time.perf_counter()
        velocity = velocity_production(result["timestamp"], result["customer_id"], result["device_id"])
        for column in velocity:
            result[column] = velocity[column].to_numpy(copy=False)
        result.attrs["velocity_generation_seconds"] = time.perf_counter() - velocity_start
    if profile in {"react2026_static_customer_history", "react2026_static_customer_history_relationships", "react2026_static_customer_history_relationships_device_global", "react2026_static_customer_history_relationships_device_global_velocity"}:
        require_columns(result, ["customer_id"], "REACT 2026 customer-history inputs")
        from .customer_history import customer_history_production
        history_start = time.perf_counter()
        history = customer_history_production(result["timestamp"], result["customer_id"], result["amount_bdt"])
        for column in history:
            result[column] = history[column].to_numpy(copy=False)
        result.attrs["customer_history_generation_seconds"] = time.perf_counter() - history_start
    return result


def validate_feature_profile(frame, config):
    if config.feature_profile not in {"react2026_static", "react2026_static_customer_history", "react2026_static_customer_history_relationships", "react2026_static_customer_history_relationships_device_global", "react2026_static_customer_history_relationships_device_global_velocity"}:
        return
    forbidden = {"transaction_id", "customer_id", "device_id", "merchant_id", "timestamp", config.target}
    if forbidden & set(config.features):
        raise ValueError("REACT 2026 static features cannot include IDs, timestamp, or target")
    for column in config.categorical_features:
        if frame[column].nunique(dropna=False) > 64:
            raise ValueError(f"REACT 2026 static categorical {column} exceeds the 64-category restriction")


def load_training(config, root):
    path = config.path(root, config.train_file)
    frame = read_table(path, id_columns=config.id_columns)
    fingerprint = {"sha256": sha256(path), "rows": len(frame), "columns": list(frame.columns)}
    frame = add_row_features(frame, config.row_features)
    held_ids = {}
    if config.feature_profile in {"react2026_static_customer_history_relationships", "react2026_static_customer_history_relationships_device_global", "react2026_static_customer_history_relationships_device_global_velocity"}:
        # Preserve OOF alignment IDs outside the state-engine working frame.
        held_ids = {column: frame.pop(column) for column in config.id_columns}
        if "merchant_id" in frame:
            frame.pop("merchant_id")
    frame = add_feature_profile(frame, config.feature_profile)
    if config.feature_profile in {"react2026_static_customer_history_relationships", "react2026_static_customer_history_relationships_device_global", "react2026_static_customer_history_relationships_device_global_velocity"}:
        # Pair state is now materialized; these high-cardinality raw IDs are
        # prohibited model inputs and are no longer needed for folds or OOF IDs.
        for column, values in held_ids.items():
            frame[column] = values
        for column in ("customer_id", "device_id"):
            if column in frame:
                frame.pop(column)
    require_columns(frame, [config.target, *config.features, *config.id_columns], "Training data")
    if frame.empty or frame[config.target].isna().any():
        raise ValueError("Training data must be nonempty with no missing targets")
    validate_feature_profile(frame, config)
    return frame, fingerprint


def load_test(config, root):
    path = config.path(root, config.test_file)
    frame = read_table(path, id_columns=config.id_columns)
    fingerprint = {"sha256": sha256(path), "rows": len(frame), "columns": list(frame.columns)}
    frame = add_feature_profile(add_row_features(frame, config.row_features), config.feature_profile)
    require_columns(frame, [*config.features, *config.id_columns], "Inference data")
    if config.target in frame.columns:
        raise ValueError("Test file contains the configured target; investigate before inference")
    validate_feature_profile(frame, config)
    return frame, fingerprint
