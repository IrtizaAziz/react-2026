"""Immutable REACT experiment runner; never exports test predictions.

An explicit user request to run an experiment is the execution authorization.
Preflight still validates the exact inputs before the next immutable ID is used.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .config import ROOT, load_config
from .ensemble import blend_experiments
from .utils import finish_record, read_json, save_json, sha256


def _next_id(root):
    used = set()
    ledger = Path(root) / "experiments/experiments.csv"
    if ledger.exists():
        import csv
        with ledger.open(newline="", encoding="utf-8") as handle:
            used.update(row["experiment_id"] for row in csv.DictReader(handle))
    for directory in (Path(root) / "outputs/reports", Path(root) / "outputs/models"):
        if directory.exists(): used.update(path.stem if path.is_file() else path.name for path in directory.iterdir())
    numbers = [int(match.group(1)) for item in used if (match := re.fullmatch(r"R(\d+)", item))]
    return f"R{max(numbers, default=0) + 1:03d}"


def _record(root, experiment):
    path = Path(root) / f"outputs/reports/{experiment}.json"
    if not path.exists(): raise FileNotFoundError(f"Missing immutable record: {experiment}")
    record = read_json(path)
    if record.get("status") != "completed": raise ValueError(f"Parent/member {experiment} is not completed")
    return record


def _require_clean_execution_config(config):
    config.validate()
    if config.predict_test or config.aggregation is not None:
        raise ValueError("Runner never permits test inference or aggregation")


def _training_preflight(root, config_path, parent, tests):
    config_path = Path(config_path); root = Path(root)
    if not config_path.exists(): raise FileNotFoundError(config_path)
    if not tests: raise ValueError("At least one explicit causal/parity test module is required")
    test_paths = [Path(test) for test in tests]
    missing = [str(path) for path in test_paths if not path.exists()]
    if missing: raise FileNotFoundError(f"Missing causal/parity tests: {missing}")
    config = load_config(config_path); _require_clean_execution_config(config)
    base = _record(root, parent); parent_config = load_config(root / f"outputs/reports/{parent}/config.json")
    prefix = config.features[:len(parent_config.features)] == parent_config.features
    same = config.features == parent_config.features
    if not (prefix or same): raise ValueError("Candidate must retain the parent's ordered manifest or exactly reuse it for a model-only comparison")
    split_path = config.path(root, config.splits_file)
    if not split_path.exists(): raise FileNotFoundError(f"Missing saved split file: {split_path}")
    split = read_json(split_path)
    if split.get("signature") != base.get("split_signature"): raise ValueError("Candidate split signature differs from parent")
    train_path = config.path(root, config.train_file)
    if sha256(train_path) != base["train_fingerprint"]["sha256"]: raise ValueError("Candidate training input hash differs from parent")
    return {"mode": "training", "config": str(config_path), "config_sha256": sha256(config_path), "parent": parent,
            "tests": [str(path) for path in test_paths], "feature_manifest": config.features,
            "feature_delta": config.features[len(parent_config.features):] if prefix else [],
            "manifest_relation": "append" if prefix and not same else "identical", "split_signature": split["signature"],
            "train_sha256": sha256(train_path), "test_inference_disabled": True}


def _blend_preflight(root, parent, members, weights):
    root = Path(root); weights = [float(x) for x in weights]
    if len(members) < 2 or len(set(members)) != len(members) or len(weights) != len(members): raise ValueError("Provide distinct members and one fixed weight each")
    if not np.isfinite(weights).all() or min(weights) < 0 or not np.isclose(sum(weights), 1, atol=1e-8, rtol=0): raise ValueError("Blend weights must be finite, nonnegative, and sum exactly to one")
    records = [_record(root, item) for item in members]
    first = records[0]; oof_details = []
    for member, record in zip(members, records):
        path = root / record["oof_path"]; meta_path = path.with_suffix(".meta.json")
        if not path.exists() or not meta_path.exists(): raise FileNotFoundError(f"Missing original OOF artifact for {member}")
        metadata = read_json(meta_path); actual = sha256(path)
        if actual != metadata.get("sha256") or metadata.get("split_signature") != first["split_signature"] or record.get("split_signature") != first["split_signature"]: raise ValueError(f"OOF provenance mismatch: {member}")
        if member != members[0]:
            for key in ("identity_hash", "data_fingerprint", "class_order", "prediction_kind", "prediction_columns", "id_columns", "split_signature"):
                if metadata[key] != oof_details[0]["metadata"][key]: raise ValueError(f"Incompatible OOF metadata: {key}")
        oof_details.append({"member": member, "path": str(path), "sha256": actual, "metadata": metadata})
    if parent not in members: raise ValueError("The comparison parent must be a blend member")
    return {"mode": "blend", "parent": parent, "members": list(members), "weights": weights, "method": "weighted",
            "split_signature": first["split_signature"], "oof": oof_details, "test_inference_disabled": True, "weight_search": False}


def preflight(root, mode, **kwargs):
    payload = _training_preflight(root, kwargs["config"], kwargs["parent"], kwargs["tests"]) if mode == "training" else _blend_preflight(root, kwargs["parent"], kwargs["members"], kwargs["weights"])
    payload["experiment_id"] = _next_id(root)
    return payload


def _test_module(path, root):
    path = Path(path).resolve()
    return str(path.relative_to(Path(root).resolve()).with_suffix("")).replace("\\", ".").replace("/", ".")


def _write_standard_report(root, experiment, parent, mode, runtime, preflight_payload):
    root = Path(root); record = _record(root, experiment); base = _record(root, parent)
    keys = {"F1": ("fold_scores", 0), "F2": ("fold_scores", 1), "mean": ("cv_mean", None), "pooled": ("pooled_covered_oof_score", None)}
    keys.update({window["name"]: ("diagnostics", window["name"]) for window in record.get("config", {}).get("diagnostic_windows", [])})
    def value(item, section, key): return item[section][key] if section == "fold_scores" else item[section] if key is None else item[section][key]["average_precision"]
    metrics = {name: float(value(record, section, key)) for name, (section, key) in keys.items() if section in record and (key is None or key in record[section])}
    deltas = {name: float(score - value(base, keys[name][0], keys[name][1])) for name, score in metrics.items() if keys[name][0] in base and (keys[name][1] is None or keys[name][1] in base[keys[name][0]])}
    recent = [deltas[name] for name in metrics if "late" in name.lower() or "july" in name.lower()]
    decision = "WIN" if deltas.get("F2", 0) >= .002 and all(delta >= 0 for delta in recent) else "LOSE" if any(delta < -.002 for delta in recent) else "FLAT-MIXED"
    report = {"experiment_id": experiment, "parent": parent, "execution_mode": mode, "preflight": preflight_payload,
              "metrics": metrics, "deltas_vs_parent": deltas, "decision": decision,
              "decision_rule": "WIN requires F2 >= +0.002 and no negative recent-period deltas; LOSE flags a recent delta below -0.002.",
              "runtime_seconds": runtime, "test_inference": False, "submission_created": False}
    save_json(root / record["provenance_path"] / "decision_report.json", report, exclusive=True)
    return report


def run(root, payload, hypothesis, change, experiment):
    root = Path(root)
    if experiment != payload["experiment_id"]: raise ValueError("Explicit experiment ID does not match this exact preflight")
    if payload["experiment_id"] != _next_id(root): raise ValueError("Experiment ID is no longer the next unused immutable ID")
    start = time.perf_counter()
    if payload["mode"] == "training":
        subprocess.run([sys.executable, "-m", "unittest", *[_test_module(path, root) for path in payload["tests"]]], cwd=root, check=True)
        from .train import train_experiment
        result = train_experiment(load_config(payload["config"]), payload["experiment_id"], hypothesis, change, root=root, parent=payload["parent"])
    else:
        result = blend_experiments(root, payload["experiment_id"], payload["members"], payload["weights"], hypothesis, change, method="weighted", predict_test=False)
        result["parent_experiment"] = payload["parent"]; finish_record(root, result)
    return _write_standard_report(root, payload["experiment_id"], payload["parent"], payload["mode"], time.perf_counter() - start, payload)


def _run_argv(args):
    """Recreate a run command exactly; supervision adds no runner behavior."""
    command = ["--root", str(args.root), "run", args.mode, "--parent", args.parent]
    for option, values in (("--config", [args.config] if args.config else []), ("--tests", args.tests),
                           ("--members", args.members), ("--weights", args.weights)):
        if values:
            command.append(option); command.extend(str(value) for value in values)
    return command + ["--experiment", args.experiment, "--hypothesis", args.hypothesis, "--change", args.change]


def supervise(root, run_argv, experiment, heartbeat_seconds=60, *, popen=subprocess.Popen,
              clock=time.monotonic, sleep=time.sleep, emit=print):
    """Run one immutable command as a child and report only supervision status."""
    if heartbeat_seconds < 1:
        raise ValueError("heartbeat_seconds must be at least one second")
    root = Path(root); process = popen([sys.executable, "-m", "src.react_runner", *run_argv], cwd=root)
    started = last_heartbeat = clock()
    emit(f"REACT supervise started: experiment={experiment} pid={process.pid}")
    while process.poll() is None:
        sleep(min(1.0, heartbeat_seconds))
        now = clock()
        if now - last_heartbeat >= heartbeat_seconds:
            emit(f"REACT supervise heartbeat: experiment={experiment} elapsed={int(now - started)}s")
            last_heartbeat = now
    exit_code = process.poll(); report_path = root / "outputs/reports" / f"{experiment}.json"
    if report_path.exists():
        record = read_json(report_path); scores = record.get("fold_scores", [])
        metrics = " ".join(f"F{index + 1}={score:.6f}" for index, score in enumerate(scores))
        training_log = root / record.get("provenance_path", f"outputs/reports/{experiment}/") / "training.log"
        emit(f"REACT supervise finished: experiment={experiment} exit_code={exit_code} status={record.get('status', 'unknown')} {metrics} report={report_path} training_log={training_log}")
    else:
        emit(f"REACT supervise finished: experiment={experiment} exit_code={exit_code} status=missing-report")
    return exit_code


def wait_for_artifact(root, relative_path, poll_seconds=25, timeout_seconds=1800, *,
                      clock=time.monotonic, sleep=time.sleep, emit=print):
    """Wait read-only for one repository-local artifact, then print a compact summary."""
    if poll_seconds <= 0 or timeout_seconds < 0:
        raise ValueError("poll_seconds must be positive and timeout_seconds must be nonnegative")
    root = Path(root).resolve(); relative_path = Path(relative_path)
    if relative_path.is_absolute():
        raise ValueError("Artifact path must be relative to the repository root")
    path = (root / relative_path).resolve()
    try:
        display_path = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Artifact path must stay within the repository root") from exc
    started = clock()
    while not path.exists():
        elapsed = clock() - started
        if elapsed >= timeout_seconds:
            emit(f"REACT wait-for timeout: path={display_path} elapsed={int(elapsed)}s")
            return 1
        sleep(min(poll_seconds, timeout_seconds - elapsed))
    details = []
    if path.suffix.lower() == ".json":
        try:
            record = read_json(path)
            for key in ("experiment_id", "status", "decision"):
                if key in record: details.append(f"{key}={record[key]}")
            metrics = record.get("metrics", {})
            scores = record.get("fold_scores", [])
            for name, index in (("F1", 0), ("F2", 1)):
                value = metrics.get(name, scores[index] if len(scores) > index else None)
                if value is not None: details.append(f"{name}={float(value):.6f}")
        except (OSError, ValueError, json.JSONDecodeError):
            details.append("json=unreadable")
    suffix = " " + " ".join(details) if details else ""
    emit(f"REACT wait-for ready: path={display_path}{suffix}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    wait = sub.add_parser("wait-for"); wait.add_argument("path")
    wait.add_argument("--poll-seconds", type=float, default=25); wait.add_argument("--timeout-seconds", type=float, default=1800)
    for name in ("preflight", "run", "supervise"):
        command = sub.add_parser(name); command.add_argument("mode", choices=("training", "blend")); command.add_argument("--parent", required=True)
        command.add_argument("--config"); command.add_argument("--tests", nargs="*"); command.add_argument("--members", nargs="+"); command.add_argument("--weights", nargs="+", type=float)
        if name in {"run", "supervise"}:
            command.add_argument("--experiment", required=True); command.add_argument("--hypothesis", required=True); command.add_argument("--change", required=True)
        if name == "supervise": command.add_argument("--heartbeat-seconds", type=float, default=60)
    args = parser.parse_args(argv); details = vars(args)
    try:
        if args.command == "preflight":
            result = preflight(args.root, args.mode, config=args.config, parent=args.parent, tests=args.tests, members=args.members, weights=args.weights)
        elif args.command == "wait-for":
            return wait_for_artifact(args.root, args.path, args.poll_seconds, args.timeout_seconds)
        elif args.command == "supervise":
            return supervise(args.root, _run_argv(args), args.experiment, args.heartbeat_seconds)
        else:
            payload = preflight(args.root, args.mode, config=args.config, parent=args.parent, tests=args.tests, members=args.members, weights=args.weights)
            result = run(args.root, payload, args.hypothesis, args.change, args.experiment)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"REACT runner stopped: {exc}\n")


if __name__ == "__main__": raise SystemExit(main())
