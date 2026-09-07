"""Freeze saved experiment source into a standalone notebook; never upload it."""
if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
from datetime import datetime, timezone
from pathlib import Path
from .config import ROOT
from .utils import checked_record, dumps, read_json, save_json, sha256

REPLAY_TOLERANCES = {"rtol": 1e-7, "atol": 1e-9, "metric_rtol": 1e-7, "metric_atol": 1e-9}


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True), "execution_count": None, "outputs": []}


def notebook(cells):
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}, "cells": cells}


REPLAY = '''import base64
import json
import os
import subprocess
import sys
from pathlib import Path

for original, replacement in INPUT_FILES.items():
    if not replacement or not Path(replacement).is_file():
        raise ValueError(f"Map organizer input {original!r} to an existing file before replay")
RUN_ROOT.mkdir(parents=True, exist_ok=False)

for item in BUNDLE["experiments"]:
    record = item["record"]
    experiment = record["experiment_id"]
    execution = RUN_ROOT / "execution_sources" / experiment
    source = execution / "src"
    source.mkdir(parents=True)
    for name, content in item["source"].items():
        (source / name).write_text(content, encoding="utf-8", newline="")
    settings = dict(record["config"])
    for key in ("train_file", "test_file", "sample_file"):
        if settings.get(key) in INPUT_FILES:
            settings[key] = str(INPUT_FILES[settings[key]])
    settings["predict_test"] = False
    saved_splits = execution / "splits.json"
    saved_splits.write_text(json.dumps(item["splits"]), encoding="utf-8")
    settings["splits_file"] = str(saved_splits)
    cfg_path = execution / "replay_config.json"
    cfg_path.write_text(json.dumps(settings), encoding="utf-8")
    item_path = execution / "expected.json"
    item_path.write_text(json.dumps(item), encoding="utf-8")
    expected_artifacts = execution / "expected_artifacts"
    expected_artifacts.mkdir()
    for name, encoded in item["artifact_snapshots"].items():
        (expected_artifacts / name).write_bytes(base64.b64decode(encoded))
    expected_submissions = execution / "expected_submissions"
    expected_submissions.mkdir()
    for name, encoded in item["submission_snapshots"].items():
        (expected_submissions / name).write_bytes(base64.b64decode(encoded))
    runner = r\'\'\'import json, sys
from pathlib import Path
from src.config import load_config
from src.train import train_experiment
from src.predict import predict_experiment
from src.ensemble import blend_experiments
from src.submission import generate_submission
from src.replay import compare_predictions, compare_submission, replay_oof_score
from src.utils import object_hash, read_json, save_json, sha256
root, cfg_path, item_path, inputs_path = map(Path, sys.argv[1:])
cfg = load_config(cfg_path)
item = read_json(item_path)
expected = item["record"]
inputs = read_json(inputs_path)
eid = expected["experiment_id"]
execution = item_path.parent
tol = item["replay_tolerances"]
if object_hash(expected["config"]) != item["config_hash"]:
    raise RuntimeError(f"{eid} frozen resolved configuration hash differs")
if object_hash(item["splits"]) != item["splits_hash"] or item["splits"].get("signature") != expected["split_signature"]:
    raise RuntimeError(f"{eid} frozen validation/fold definition differs")
for name, expected_hash in expected["source_hashes"].items():
    if sha256(Path(__file__).parent / "src" / name) != expected_hash:
        raise RuntimeError(f"{eid} saved source snapshot differs: {name}")
if expected.get("ensemble"):
    recipe = expected["ensemble"]
    result = blend_experiments(root, eid, recipe["members"], recipe["weights"], expected["hypothesis"], expected["change_description"], method=recipe["method"], predict_test=bool(expected.get("prediction_path")), train_file=cfg.train_file)
else:
    result = train_experiment(cfg, eid, expected["hypothesis"], expected["change_description"], root=root, parent=expected.get("parent_experiment"))
    if expected.get("prediction_path"):
        original_test = expected.get("inference", {}).get("test_file") or expected["config"]["test_file"]
        result = predict_experiment(root, eid, test_file=inputs[original_test], aggregation="mean")
if result["train_fingerprint"] != expected["train_fingerprint"] or result["split_signature"] != expected["split_signature"]:
    raise RuntimeError(f"{eid} input fingerprint or validation definition differs")
report = {"experiment_id": eid, "tolerances": tol, "artifacts": {}}
for key in item["artifact_hashes"]:
    comparison = compare_predictions(execution / "expected_artifacts" / item["artifact_files"][key], root / result[key], rtol=tol["rtol"], atol=tol["atol"])
    report["artifacts"][key] = comparison
    if not comparison["reproduced"]:
        raise RuntimeError(f"{eid} {key} is not numerically reproduced: {comparison}")
metric_report = replay_oof_score(expected, result, root, cfg, rtol=tol["metric_rtol"], atol=tol["metric_atol"])
report["oof_metric"] = metric_report
if not metric_report["within_tolerance"]:
    raise RuntimeError(f"{eid} replayed OOF metric differs from recorded CV score: {metric_report}")
for entry in expected.get("submissions", []):
    for key, value in entry["config"].items():
        setattr(cfg, key, inputs.get(value, value) if key == "sample_file" else value)
    # Preserve the recorded descriptive filename by recovering its description.
    filename = Path(entry["submission_path"]).name
    prefix = f"sub_{eid}_{expected['model']}_"
    description = filename[len(prefix):].rsplit("_cv", 1)[0]
    output = generate_submission(root, eid, config=cfg, description=description)
    comparison = compare_submission(execution / "expected_submissions" / filename, output, id_columns=cfg.id_columns, rtol=tol["rtol"], atol=tol["atol"])
    report["artifacts"][f"submission:{filename}"] = comparison
    if not comparison["reproduced"]:
        raise RuntimeError(f"Submission is not numerically reproduced: {filename}: {comparison}")
save_json(execution / "replay_report.json", report, exclusive=True)
print(f"VERIFIED: {eid}, exact provenance + numerical prediction replay; report: {execution / 'replay_report.json'}")
\'\'\'
    runner_path = execution / "run_replay.py"
    runner_path.write_text(runner, encoding="utf-8")
    input_map_path = execution / "inputs.json"
    input_map_path.write_text(json.dumps({k: str(v) for k, v in INPUT_FILES.items()}), encoding="utf-8")
    process = subprocess.run([sys.executable, str(runner_path), str(RUN_ROOT), str(cfg_path), str(item_path), str(input_map_path)], cwd=execution, text=True, capture_output=True)
    print(process.stdout)
    if process.returncode:
        print(process.stderr)
        raise RuntimeError(f"Replay failed for {experiment}; fix the reported cause before claiming reproducibility")
print("All bundled experiments verified. Submission generation did not upload anything.")
'''


def export_notebook(root, experiment, *, output=None):
    root = Path(root)
    ordered, seen, visiting = [], set(), set()
    def visit(eid):
        if eid in seen:
            return
        if eid in visiting:
            raise ValueError("Ensemble provenance contains a cycle")
        visiting.add(eid)
        record = checked_record(root, eid)
        for member in record.get("ensemble", {}).get("members", []):
            visit(member)
        provenance = root / record["provenance_path"]
        sources = {}
        for name, expected_hash in record["source_hashes"].items():
            if Path(name).name != name or not name.endswith(".py"):
                raise ValueError("Invalid source snapshot filename")
            path = provenance / "source" / name
            if sha256(path) != expected_hash:
                raise ValueError(f"Source snapshot was modified: {eid}/{name}")
            sources[name] = path.read_bytes().decode("utf-8")
        artifacts = {}
        artifact_snapshots, artifact_files = {}, {}
        for key in ("oof_path", "prediction_path"):
            if record.get(key):
                path = root / record[key]
                metadata = read_json(path.with_suffix(".meta.json"))
                if sha256(path) != metadata["sha256"]:
                    raise ValueError(f"Artifact was modified: {path}")
                artifacts[key] = metadata["sha256"]
                bundled = f"{key}-{path.name}"
                artifact_files[key] = bundled
                artifact_snapshots[bundled] = __import__("base64").b64encode(path.read_bytes()).decode("ascii")
                artifact_snapshots[Path(bundled).with_suffix(".meta.json").name] = __import__("base64").b64encode(path.with_suffix(".meta.json").read_bytes()).decode("ascii")
        submission_snapshots = {}
        for entry in record.get("submissions", []):
            path = root / entry["submission_path"]
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Submission artifact was modified: {path}")
            submission_snapshots[path.name] = __import__("base64").b64encode(path.read_bytes()).decode("ascii")
        ordered.append({"record": record, "source": sources, "splits": read_json(root / record["splits_path"]),
                        "environment": read_json(provenance / "environment.json"), "artifact_hashes": artifacts,
                        "artifact_snapshots": artifact_snapshots, "artifact_files": artifact_files, "submission_snapshots": submission_snapshots,
                        "config_hash": __import__("hashlib").sha256(dumps(record["config"]).encode()).hexdigest(),
                        "splits_hash": __import__("hashlib").sha256(dumps(read_json(root / record["splits_path"])).encode()).hexdigest(),
                        "replay_tolerances": REPLAY_TOLERANCES})
        visiting.remove(eid)
        seen.add(eid)
    visit(experiment)
    inputs = {}
    for item in ordered:
        record = item["record"]
        inputs[record["config"]["train_file"]] = None
        if record.get("prediction_path") and not record.get("ensemble"):
            inputs[record.get("inference", {}).get("test_file") or record["config"]["test_file"]] = None
        for entry in record.get("submissions", []):
            inputs[entry["config"]["sample_file"]] = None
    bundle = {"candidate": experiment, "experiments": ordered}
    cells = [markdown(f"# REACT 2026 — reproduce {experiment}\n\nThis notebook embeds the saved source for each experiment. It loads only mapped organizer files. Run from a fresh kernel and a new output directory. No Kaggle upload is implemented.\n\nFor a serious candidate: commit local code, run this notebook, verify every artifact, Save Version, then record and privately share that exact version with the organizers. Do not share with other teams."),
             markdown("## Runtime and disclosures\n\nThe frozen bundle records Python, package versions, source hashes, model parameters, seeds, and exact folds. Inspect `BUNDLE['experiments'][i]['environment']` and install the recorded required packages explicitly if needed. Optional model libraries are not installed automatically. Numerical differences cause replay verification to stop.\n\nPretrained models: none in the starter. If activated later, record source, exact public model name/revision, and loading code directly in this notebook and the method summary."),
             code("import json\nfrom pathlib import Path\nBUNDLE = json.loads(" + repr(dumps(bundle)) + ")\nprint([(e['record']['experiment_id'], e['environment']['python']) for e in BUNDLE['experiments']])\n"),
             markdown("## Map competition inputs\n\nReplace each `None` with its `/kaggle/input/...` path. These are paths to the same organizer files, not new data. Choose a fresh writable RUN_ROOT for every execution. The original input paths are provenance labels only."),
             code("RUN_ROOT = Path(globals().get('REPLAY_ROOT', '/kaggle/working/react-replay-" + experiment + "')).resolve()\nINPUT_FILES = globals().get('REPLAY_INPUTS', json.loads(" + repr(dumps(inputs)) + "))\n"),
             code(REPLAY),
             markdown("## Commit and record the exact version\n\nConfirm all VERIFIED messages and generated CSVs under RUN_ROOT/submissions. Save a committed Kaggle Notebook version that runs training and inference. Add its exact version URL to experiments/experiments.csv, the candidate report, and submission sidecar. Upload a CSV only on explicit human instruction. Up to two final selections remain human decisions. Never replace a prior candidate or infer a successful replay from a completed training cell alone.")]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path(output) if output else root / "notebooks" / f"reproduce_{experiment}_{stamp}.ipynb"
    save_json(destination, notebook(cells), exclusive=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        print(export_notebook(args.root, args.experiment, output=args.output))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Export stopped: {exc}\n")


if __name__ == "__main__":
    main()
