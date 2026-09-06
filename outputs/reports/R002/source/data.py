"""Readers never discover or load test data as a side effect of training."""
from pathlib import Path
import numpy as np
import pandas as pd
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
    result = frame.copy()
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


def add_feature_profile(frame, profile):
    """Materialize approved deterministic features without observing other rows."""
    if profile is None:
        return frame
    if profile != "react2026_static":
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
    return result


def validate_feature_profile(frame, config):
    if config.feature_profile != "react2026_static":
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
    frame = add_feature_profile(add_row_features(frame, config.row_features), config.feature_profile)
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
