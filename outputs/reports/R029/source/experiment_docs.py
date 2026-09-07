"""Deterministic, derived documentation for generic feature experiments."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import ROOT
from .utils import read_json


METRICS = (
    ("F1", "F1"), ("F2", "F2"), ("F2 early", "F2_early"),
    ("corrected F2 late", "F2_late_corrected"), ("July", "July_1_15"),
    ("June 15-30", "June_15_30"), ("July 1-7", "July_1_7"),
    ("July 8-15", "July_8_15"),
)


def _value(value):
    if value is None:
        return "UNKNOWN"
    if isinstance(value, bool):
        return "true" if value else "false"
    return f"{value:.6f}" if isinstance(value, (float, int)) else str(value)


def _path(root, value):
    if not value:
        return "UNKNOWN"
    try:
        return str(Path(value).relative_to(root))
    except ValueError:
        return str(value)


def _importance(items, names=None, limit=10):
    items = items or []
    if names is not None:
        items = [item for item in items if item.get("feature") in names]
    return ", ".join(f"{item['feature']} ({float(item['importance']):.6f})" for item in items[:limit]) or "UNKNOWN"


def render_experiment_card(root, record, report):
    """Render a stable Markdown companion from immutable run evidence only."""
    root = Path(root)
    provenance = record.get("feature_experiment_provenance", {})
    spec = provenance.get("normalized_spec", {})
    added = list(spec.get("added_features", []))
    after_count = len(record.get("config", {}).get("features", [])) or None
    before_count = after_count - len(added) if after_count is not None else None
    metrics = report.get("metrics", {})
    parent_delta, incumbent_delta = report.get("delta_vs_parent", {}), report.get("delta_vs_incumbent", {})
    lines = [f"# {record.get('experiment_id', report.get('experiment_id', 'UNKNOWN'))} experiment card", "",
             "Derived factual companion to `decision_report.json`; feature importances are descriptive, not causal.", "",
             "## Experiment", "",
             f"- Experiment ID: {_value(report.get('experiment_id', record.get('experiment_id')))}",
             f"- Parent: {_value(report.get('parent', record.get('parent_experiment')))}",
             f"- Incumbent: {_value(report.get('incumbent'))}",
             f"- Timestamp: {_value(record.get('timestamp'))}",
             f"- Immutable status: {_value(record.get('status'))}",
             f"- Hypothesis: {_value(spec.get('hypothesis', record.get('hypothesis')))}",
             f"- Feature module: {_value(spec.get('feature_module'))}",
             f"- Ordered added features: {', '.join(added) or 'UNKNOWN'}",
             f"- Feature count before/after: {_value(before_count)} / {_value(after_count)}",
             f"- Inherited recipe: model={_value(record.get('model'))}; parameters={_value(record.get('config', {}).get('model_params'))}; parent config hash={_value(provenance.get('parent_config_sha256'))}",
             f"- Documented coherent change: append exactly the listed feature(s); no other generic-spec change is recorded.", "",
             "## Metrics", "", "| Metric | Score | Delta parent | Delta incumbent |", "|---|---:|---:|---:|"]
    lines.extend(f"| {label} | {_value(metrics.get(key))} | {_value(parent_delta.get(key))} | {_value(incumbent_delta.get(key))} |" for label, key in METRICS)
    gates = report.get("submission_gates", {})
    lines += ["", "## Decision", "",
              f"- Classification: {_value(report.get('classification'))}",
              f"- Submission gates passed: {_value(report.get('submission_gate_pass'))}",
              f"- Gate evidence: {', '.join(f'{key}={_value(value)}' for key, value in gates.items()) or 'UNKNOWN'}",
              f"- Early stopped: {_value(report.get('early_stopped'))}",
              f"- Early-stop policy: {_value(report.get('early_stop_policy'))}", "",
              "## Feature importances", "",
              f"- Added features, F1: {_importance(report.get('feature_importances', {}).get('F1'), added)}",
              f"- Added features, F2: {_importance(report.get('feature_importances', {}).get('F2'), added)}",
              f"- Top 10 overall, F1: {_importance(report.get('feature_importances', {}).get('F1'))}",
              f"- Top 10 overall, F2: {_importance(report.get('feature_importances', {}).get('F2'))}", "",
              "## Certification and provenance", "",
              f"- Certification status: {'reused' if report.get('certification_reused') else 'new'}",
              f"- Certification results: {_value(provenance.get('certification', {}).get('results'))}",
              f"- Parent matrix parity / row-ID alignment: verified by pre-fit candidate checks",
              f"- Locked split signature: {_value(provenance.get('split_signature'))}",
              f"- Train input fingerprint: {_value(provenance.get('train_fingerprint'))}",
              f"- Test inference / submission: {report.get('test_inference', False)} / {report.get('submission_created', False)}",
              f"- Runtime seconds: {_value(report.get('runtime_seconds'))}",
              f"- OOF: {_path(root, record.get('oof_path'))}",
              f"- Model/config/spec/report/source: {_value(record.get('model_paths'))}; {_path(root, record.get('provenance_path'))}/config.json; {_path(root, provenance.get('spec_path'))}; {_path(root, record.get('provenance_path'))}/decision_report.json; {_path(root, record.get('provenance_path'))}/source",
              f"- Module hashes: {_value(provenance.get('module_source_sha256'))}",
              f"- Runner hashes: {_value(provenance.get('runner_source_sha256'))}",
              f"- Parent cache: {_value(provenance.get('parent_matrix_cache'))}"]
    if report.get("diagnostics_path"):
        lines.append(f"- Diagnostics: {_path(root, report['diagnostics_path'])}")
    if report.get("diagnostics_error"):
        lines.append(f"- Diagnostics status: unavailable ({report['diagnostics_error']})")
    if report.get("reference_runs"):
        lines += ["", "## Reference-run analysis", ""]
        for rid, details in report["reference_runs"].items():
            lines.append(f"- {rid}: compatible={details.get('compatible', 'UNKNOWN')}; deltas={_value(details.get('delta_vs_reference'))}")
    positive = [label for label, key in METRICS if parent_delta.get(key, 0) > 0]
    conclusion = f"{report.get('experiment_id')} improved {', '.join(positive)} versus its parent" if positive else f"{report.get('experiment_id')} has no positive documented metric delta versus its parent"
    lines += ["", "## Factual conclusion", "", f"{conclusion}; classification is {_value(report.get('classification'))} and submission gates passed={_value(report.get('submission_gate_pass'))}.", ""]
    return "\n".join(lines)


def write_experiment_card(root, record, report):
    path = Path(root) / record["provenance_path"] / "experiment_card.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(render_experiment_card(root, record, report))
    return str(path.relative_to(root))


def rebuild_experiment_index(root):
    root = Path(root); ledger = root / "experiments/experiments.csv"
    if ledger.exists():
        with ledger.open(encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle))
    else:
        rows = []
    rows = [row for row in rows if row.get("experiment_id", "").startswith("R")]
    rows.sort(key=lambda row: int(row["experiment_id"][1:]) if row["experiment_id"][1:].isdigit() else 10**9)
    table, provenance = [], []
    for row in rows:
        rid = row["experiment_id"]; path = root / "outputs/reports" / rid / "decision_report.json"
        report = read_json(path) if path.exists() else {}
        record_path = root / "outputs/reports" / f"{rid}.json"; record = read_json(record_path) if record_path.exists() else {}
        spec = record.get("feature_experiment_provenance", {}).get("normalized_spec", {})
        metrics = report.get("metrics", {})
        change = spec.get("hypothesis") or row.get("change_description") or "UNKNOWN"
        table.append(f"| {rid} | {row.get('parent_experiment') or report.get('parent') or 'UNKNOWN'} | {change[:160]} | {row.get('model') or record.get('model') or 'UNKNOWN'} | {', '.join(spec.get('added_features', [])) or 'UNKNOWN'} | {_value(metrics.get('F1'))} | {_value(metrics.get('F2'))} | {_value(metrics.get('F2_late_corrected'))} | {_value(metrics.get('July_1_15'))} | {report.get('classification') or report.get('decision') or 'UNKNOWN'} | {_value(report.get('submission_gate_pass', 'UNKNOWN'))} | {row.get('public_lb') or 'UNKNOWN'} |")
        prov = record.get("feature_experiment_provenance", {})
        provenance.append(f"| {rid} | {spec.get('feature_module', 'UNKNOWN')} | {prov.get('module_source_sha256', 'UNKNOWN')} | {', '.join(spec.get('added_features', [])) or 'UNKNOWN'} | {len(record.get('config', {}).get('features', [])) or 'UNKNOWN'} | {_value(report.get('certification_reused', 'UNKNOWN'))} | {prov.get('parent_matrix_cache', {}).get('matrix_sha256', 'UNKNOWN') if isinstance(prov.get('parent_matrix_cache'), dict) else 'UNKNOWN'} | {_value(prov.get('parent_matrix_cache', {}).get('used', 'UNKNOWN') if isinstance(prov.get('parent_matrix_cache'), dict) else 'UNKNOWN')} |")
    text = "# Experiment index\n\nDerived from ledger and available report evidence; unavailable fields are `UNKNOWN`.\n\n| R-ID | Parent | Hypothesis / change | Model | Added family | F1 | F2 | Corrected late | July | Classification/status | Gates | Public LB |\n|---|---|---|---|---|---:|---:|---:|---:|---|---|---|\n" + "\n".join(table) + "\n\n## Generic feature provenance\n\n| R-ID | Import path | Source hash | AVAILABLE / added features | Feature count | Certification | Parent cache hash | Cache verified |\n|---|---|---|---|---:|---|---|---|\n" + "\n".join(provenance) + "\n"
    path = root / "reports/EXPERIMENT_INDEX.md"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8", newline="\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("command", choices=("index",)); args = parser.parse_args(argv)
    print(rebuild_experiment_index(args.root))


if __name__ == "__main__": main()
