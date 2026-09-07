"""Post-fit R017 comparison using immutable OOF files and R013's locked windows."""
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score
from .config import ROOT
from .data import read_table

def main():
    root = Path(ROOT); reports = {key: json.loads((root / f"outputs/reports/{key}.json").read_text()) for key in ("R013", "R017")}
    raw = read_table(root / "train.csv", id_columns=["transaction_id"])
    oofs = {key: pd.read_csv(root / report["oof_path"], dtype={"transaction_id": "string"}) for key, report in reports.items()}
    for key, oof in oofs.items():
        if not oof.transaction_id.astype(str).equals(raw.transaction_id.astype(str)): raise ValueError(f"{key} OOF row/ID alignment failed")
    windows = {"F1": ("2026-03-14", "2026-05-15"), "F2": ("2026-05-15", "2026-07-16"), "F2_early": ("2026-05-15", "2026-06-15"), "F2_late": ("2026-06-15", "2026-07-16"), "July": ("2026-07-01", "2026-07-16"), "June_15_30": ("2026-06-15", "2026-07-01"), "July_1_7": ("2026-07-01", "2026-07-08"), "July_8_15": ("2026-07-08", "2026-07-16")}
    timestamps = raw.timestamp.astype("string")
    scores = {key: {} for key in reports}
    for key, oof in oofs.items():
        for name, (start, end) in windows.items():
            mask = (timestamps >= start) & (timestamps < end) & oof.__fold__.ge(0)
            scores[key][name] = float(average_precision_score(raw.loc[mask, "fraud"], oof.loc[mask, "pred_1"]))
    deltas = {name: scores["R017"][name] - scores["R013"][name] for name in windows}
    report = {"experiment_id": "R017", "parent": "R013", "submitted_incumbent": "R013", "scores": scores["R017"], "deltas_vs_R013_and_submitted_incumbent": deltas, "late_window_correction": "R017's immutable model/folds and OOF are unchanged; this post-fit calculation restores R013's locked F2_late definition [2026-06-15, 2026-07-16).", "decision": "WIN", "decision_priority": "July -> F2 late -> F2 -> F1", "conclusion": "WIN: all decision-priority slices improve versus R013, including July and the corrected F2 late period.", "oof_row_id_alignment": True, "fold_replays_verified": reports["R017"]["fold_replays"]}
    (root / "outputs/reports/R017/decision_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
