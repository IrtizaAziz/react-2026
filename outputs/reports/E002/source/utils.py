"""Provenance, exclusive artifact creation, and a deliberately small CSV ledger."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
from datetime import datetime, timezone

TRACKER_COLUMNS = "experiment_id timestamp parent_experiment model hypothesis change_description validation_method n_folds seed cv_mean cv_std fold_scores public_lb private_lb parameters oof_path prediction_path submission_path git_commit status conclusion notes training_seconds provenance_path kaggle_notebook_version".split()


def now():
    return datetime.now(timezone.utc).isoformat()


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def dumps(value):
    return json.dumps(value, indent=2, default=json_default, allow_nan=False)


def save_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = dumps(value)
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
    else:
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def seed_everything(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def git_info(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
        return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None,
                "git_note": None if commit else "No committed Git version; commit before a serious candidate."}
    except FileNotFoundError:
        return {"git_commit": None, "git_dirty": None, "git_note": "Git is unavailable."}


def snapshot_source(destination):
    source = Path(__file__).resolve().parent
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in sorted(source.glob("*.py")):
        content = path.read_bytes()
        (destination / path.name).write_bytes(content)
        hashes[path.name] = sha256(path)
    return hashes


def environment(*, used_packages=()):
    import platform
    packages = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}
    core = {name: packages.get(name) for name in ("numpy", "pandas", "scikit-learn")}
    optional = {name: packages.get(name) for name in used_packages if name in {"catboost", "lightgbm", "xgboost"} and packages.get(name)}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "core_packages": core, "used_optional_packages": optional}


def update_tracker(root, record):
    path = Path(root) / "experiments/experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("Tracker is being edited by another process; do not run concurrent experiments.") from exc
    try:
        os.close(descriptor)
        rows = []
        fields = TRACKER_COLUMNS.copy()
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields += [f for f in (reader.fieldnames or []) if f not in fields]
                rows = list(reader)
        row = next((r for r in rows if r["experiment_id"] == record["experiment_id"]), None)
        if row is None:
            row = {"experiment_id": record["experiment_id"]}
            rows.append(row)
        for key in fields:
            if key in record:
                value = record[key]
                row[key] = json.dumps(value, default=json_default, allow_nan=False) if isinstance(value, (dict, list)) else value
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        lock.unlink()


def report_path(root, experiment):
    if not re.fullmatch(r"(?:E\d{3,}|SMOKE\d{3,})", experiment):
        raise ValueError("Use a unique ID such as E001 (SMOKE001 is reserved for isolated tests)")
    return Path(root) / "outputs/reports" / f"{experiment}.json"


def reserve_experiment(root, experiment, config, hypothesis, change, parent=None):
    if not hypothesis.strip() or not change.strip():
        raise ValueError("A hypothesis and change description are required")
    if experiment.startswith("SMOKE") != config.smoke_test:
        raise ValueError("SMOKE IDs require smoke_test=true; competition runs require E IDs")
    path = report_path(root, experiment)
    ledger = Path(root) / "experiments/experiments.csv"
    if ledger.exists():
        with ledger.open(newline="", encoding="utf-8") as handle:
            if any(row["experiment_id"] == experiment for row in csv.DictReader(handle)):
                raise FileExistsError(f"Experiment {experiment} already appears in the ledger; IDs are never reused")
    for artifact in (Path(root) / "outputs/models" / experiment,
                     Path(root) / "outputs/reports" / experiment,
                     Path(root) / "outputs/oof" / f"{experiment}.csv",
                     Path(root) / "outputs/predictions" / f"{experiment}.csv"):
        if artifact.exists():
            raise FileExistsError(f"Existing artifact reserves this experiment ID: {artifact}")
    record = {"experiment_id": experiment, "timestamp": now(), "status": "running",
              "hypothesis": hypothesis, "change_description": change, "parent_experiment": parent,
              "model": config.model, "config": config.to_dict(), "parameters": config.model_params,
              "validation_method": config.validation_type, "n_folds": config.n_splits, "seed": config.seed,
              "provenance_path": f"outputs/reports/{experiment}/", **git_info(root)}
    save_json(path, record, exclusive=True)
    update_tracker(root, record)
    return record


def finish_record(root, record):
    save_json(report_path(root, record["experiment_id"]), record)
    update_tracker(root, record)


def save_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def checked_record(root, experiment):
    record = read_json(report_path(root, experiment))
    if record["status"] != "completed":
        raise ValueError(f"{experiment} is not completed")
    return record
