"""Rank completed experiments and measure OOF prediction diversity."""
if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Config, ROOT
from .predict import read_predictions
from .utils import read_json


def _records(root):
    path = Path(root) / "experiments/experiments.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def comparison(root=ROOT, *, prefix="R"):
    """Compare one experiment namespace and one exact validation/data contract."""
    root = Path(root)
    rows, completed = [], []
    for row in _records(root):
        if prefix and not row.get("experiment_id", "").startswith(prefix):
            continue
        if row.get("status") not in {"completed", "candidate", "promoted", "submitted", "selected", "rejected"} or not row.get("cv_mean"):
            continue
        record = read_json(root / "outputs/reports" / f"{row['experiment_id']}.json")
        config = Config(**record["config"])
        compatibility = {"task": config.task, "metric": config.metric, "metric_direction": config.metric_direction,
                         "class_order": config.class_order, "target": config.target,
                         "validation_type": config.validation_type,
                         "train_fingerprint": record.get("train_fingerprint"),
                         "split_signature": record.get("split_signature")}
        rows.append({"experiment": row["experiment_id"], "model": row.get("model", ""),
                     "features": ",".join(config.features or []), "cv_mean": float(row["cv_mean"]),
                     "cv_std": float(row.get("cv_std") or 0), "runtime_s": float(row.get("training_seconds") or 0),
                     "screening_score": record.get("screening_score"), "status": row.get("status", ""),
                     "compatibility": compatibility})
        completed.append((record, config, compatibility))
    if not rows:
        return pd.DataFrame(columns=["experiment", "model", "features", "cv_mean", "cv_std", "screening_score", "runtime_s", "difference_from_best", "status"]), {}
    baseline = completed[0][2]
    compatible = [(row, entry) for row, entry in zip(rows, completed) if entry[2] == baseline]
    rows = [row for row, _ in compatible]
    completed = [entry for _, entry in compatible]
    higher = completed[0][1].metric_direction == "higher"
    best = max(r["cv_mean"] for r in rows) if higher else min(r["cv_mean"] for r in rows)
    for row in rows:
        row["difference_from_best"] = row["cv_mean"] - best
    table = pd.DataFrame(rows).sort_values("cv_mean", ascending=not higher, ignore_index=True)
    diagnostics = {}
    for record, config, _ in completed:
        if not record.get("oof_path"):
            continue
        frame, values, _ = read_predictions(root / record["oof_path"])
        values = np.asarray(values, dtype=float)
        if config.prediction_kind == "probability":
            values = values[:, config.class_order.index(config.positive_class)]
        mask = np.isfinite(values)
        diagnostics[record["experiment_id"]] = {"oof_rows": len(frame), "covered_rows": int(mask.sum()),
            "prediction_mean": float(values[mask].mean()) if mask.any() else None,
            "prediction_std": float(values[mask].std()) if mask.any() else None,
            "finite_on_covered_rows": bool(mask.any() and np.isfinite(values[mask]).all())}
    return table, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prefix", default="R", choices=["R", "E", "SMOKE"],
                        help="Compare one namespace only; live R runs are the default.")
    args = parser.parse_args()
    table, _ = comparison(args.root, prefix=args.prefix)
    if table.empty:
        print("No completed experiments found.")
    else:
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
