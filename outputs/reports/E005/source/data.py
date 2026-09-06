"""Readers never discover or load test data as a side effect of training."""
from pathlib import Path
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
    require_columns(frame, ["SibSp", "Parch"], "Row-local feature inputs")
    result = frame.copy()
    family_size = result["SibSp"] + result["Parch"] + 1
    if "FamilySize" in row_features:
        result["FamilySize"] = family_size
    if "IsAlone" in row_features:
        result["IsAlone"] = (family_size == 1).astype(int)
    return result


def load_training(config, root):
    path = config.path(root, config.train_file)
    frame = read_table(path, id_columns=config.id_columns)
    fingerprint = {"sha256": sha256(path), "rows": len(frame), "columns": list(frame.columns)}
    frame = add_row_features(frame, config.row_features)
    require_columns(frame, [config.target, *config.features, *config.id_columns], "Training data")
    if frame.empty or frame[config.target].isna().any():
        raise ValueError("Training data must be nonempty with no missing targets")
    return frame, fingerprint


def load_test(config, root):
    path = config.path(root, config.test_file)
    frame = read_table(path, id_columns=config.id_columns)
    fingerprint = {"sha256": sha256(path), "rows": len(frame), "columns": list(frame.columns)}
    frame = add_row_features(frame, config.row_features)
    require_columns(frame, [*config.features, *config.id_columns], "Inference data")
    if config.target in frame.columns:
        raise ValueError("Test file contains the configured target; investigate before inference")
    return frame, fingerprint
