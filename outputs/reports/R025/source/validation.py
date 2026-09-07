"""Explicit validation only. No dataset heuristic selects a splitter."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn import model_selection
from .utils import object_hash, read_json, save_json


def split_settings(config):
    keys = ["validation_type", "validation_rationale", "n_splits", "shuffle", "seed",
            "group_column", "time_column", "time_gap", "time_valid_fraction", "calendar_folds", "target"]
    return {key: getattr(config, key) for key in keys}


def make_splits(frame, config, fingerprint, *, save_to=None, reuse=None):
    kind, n = config.validation_type, len(frame)
    if not kind or not config.n_splits:
        raise ValueError("Validation strategy and n_splits must be explicitly configured")
    settings = split_settings(config)
    groups = None
    if kind in {"group", "stratified_group"}:
        if not config.group_column or config.group_column not in frame:
            raise ValueError("Group validation requires an existing group_column")
        groups = frame[config.group_column]
        if groups.isna().any() or groups.nunique() < config.n_splits:
            raise ValueError("Groups must be nonmissing with at least n_splits unique groups")
    y = frame[config.target]
    if kind in {"stratified", "stratified_group"}:
        if config.task != "classification" or y.value_counts().min() < config.n_splits:
            raise ValueError("Stratification requires classification and at least n_splits rows per class")
    times = None
    if kind in {"time", "time_holdout", "calendar_time"}:
        if config.shuffle or not config.time_column or config.time_column not in frame:
            raise ValueError("Time validation requires time_column and shuffle=false")
        values = frame[config.time_column]
        times = values if pd.api.types.is_numeric_dtype(values) else pd.to_datetime(values, errors="raise", utc=True, format="mixed")
        if times.isna().any():
            raise ValueError("Time column contains missing values")
        differences = times.sort_values().diff().dropna()
        if kind in {"time", "time_holdout"} and differences.nunique() > 1:
            warnings.warn("Irregular timestamps: TimeSeriesSplit uses row counts, not elapsed durations.")
        if kind in {"time", "time_holdout"} and times.duplicated().any():
            warnings.warn("Repeated timestamps detected; tied train/validation boundaries will be rejected.")
    if reuse:
        payload = read_json(reuse)
        if payload["data_fingerprint"] != fingerprint or payload["settings"] != settings:
            raise ValueError("Saved splits do not match this dataset/order or validation configuration")
        pairs = [(np.array(p["train"], dtype=int), np.array(p["valid"], dtype=int)) for p in payload["splits"]]
    else:
        common = {"n_splits": config.n_splits}
        random = {"shuffle": config.shuffle, "random_state": config.seed if config.shuffle else None}
        if kind == "kfold":
            splitter = model_selection.KFold(**common, **random)
        elif kind == "stratified":
            splitter = model_selection.StratifiedKFold(**common, **random)
        elif kind == "group":
            if config.shuffle:
                raise ValueError("This portable GroupKFold adapter requires shuffle=false")
            splitter = model_selection.GroupKFold(**common)
        elif kind == "stratified_group":
            cls = getattr(model_selection, "StratifiedGroupKFold", None)
            if cls is None:
                raise ImportError("Installed scikit-learn lacks StratifiedGroupKFold; no fallback was selected")
            splitter = cls(**common, **random)
        elif kind == "time":
            splitter = model_selection.TimeSeriesSplit(**common, gap=config.time_gap)
        elif kind == "time_holdout":
            if config.n_splits != 1:
                raise ValueError("time_holdout requires n_splits=1")
            ordered = np.argsort(times.to_numpy(), kind="stable")
            valid_size = max(1, int(np.ceil(len(ordered) * config.time_valid_fraction)))
            valid_start = len(ordered) - valid_size
            train_end = valid_start - config.time_gap
            if train_end < 1:
                raise ValueError("time_holdout leaves no training rows after the configured gap")
            pairs = [(ordered[:train_end], ordered[valid_start:])]
            splitter = None
        elif kind == "calendar_time":
            pairs = []
            for fold in config.calendar_folds:
                train_before = pd.Timestamp(fold["train_before"], tz="UTC")
                valid_start = pd.Timestamp(fold["valid_start"], tz="UTC")
                valid_end = pd.Timestamp(fold["valid_end"], tz="UTC")
                if not train_before <= valid_start < valid_end:
                    raise ValueError("Calendar fold bounds must satisfy train_before <= valid_start < valid_end")
                train = np.flatnonzero((times < train_before).to_numpy())
                valid = np.flatnonzero(((times >= valid_start) & (times < valid_end)).to_numpy())
                if not len(train) or not len(valid):
                    raise ValueError("Calendar fold leaves empty training or validation data")
                pairs.append((train, valid))
            splitter = None
        else:
            raise ValueError("validation_type must be kfold, stratified, group, stratified_group, time, time_holdout, or calendar_time")
        if kind not in {"time_holdout", "calendar_time"}:
            order = np.argsort(times.to_numpy(), kind="stable") if times is not None else np.arange(n)
            pairs = [(order[a], order[b]) for a, b in splitter.split(frame.iloc[order], y.iloc[order], groups)]
    if len(pairs) != config.n_splits:
        raise ValueError("Saved split count differs from configured n_splits")
    assignment = np.full(n, -1, dtype=int)
    for fold, (train, valid) in enumerate(pairs):
        for indices in (train, valid):
            if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= n:
                raise ValueError("Invalid or duplicated split indices")
        if np.intersect1d(train, valid).size or (assignment[valid] != -1).any():
            raise ValueError("Overlapping training/validation indices or repeated OOF assignment")
        if groups is not None and set(groups.iloc[train]) & set(groups.iloc[valid]):
            raise ValueError("Groups cross the train/validation boundary")
        if times is not None and times.iloc[train].max() >= times.iloc[valid].min():
            raise ValueError("Temporal overlap or tied timestamp at fold boundary; decide a suitable time/group strategy")
        if config.task == "classification" and set(y.iloc[train]) != set(config.class_order):
            raise ValueError(f"Fold {fold} training data lacks a configured class; revise validation explicitly")
        assignment[valid] = fold
    if kind not in {"time", "time_holdout", "calendar_time"} and (assignment == -1).any():
        raise ValueError("Non-temporal CV must cover every row exactly once")
    payload = {"settings": settings, "data_fingerprint": fingerprint,
               "splits": [{"train": a.tolist(), "valid": b.tolist(),
                           "train_rows": int(len(a)), "valid_rows": int(len(b)),
                           "train_time_min": str(times.iloc[a].min()) if times is not None else None,
                           "train_time_max": str(times.iloc[a].max()) if times is not None else None,
                           "valid_time_min": str(times.iloc[b].min()) if times is not None else None,
                           "valid_time_max": str(times.iloc[b].max()) if times is not None else None}
                          for a, b in pairs],
               "fold_assignment": assignment.tolist()}
    payload["signature"] = object_hash(payload)
    if reuse and payload["signature"] != read_json(reuse).get("signature"):
        raise ValueError("Saved split integrity check failed")
    if save_to:
        save_json(save_to, payload, exclusive=True)
    return pairs, assignment, payload
