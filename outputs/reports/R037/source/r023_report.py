"""Post-fit R023 decision and descriptive support report from immutable OOFs."""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from .config import ROOT, load_config
from .data import load_training, read_table


def _score(raw, oof, start, end):
    mask = ((raw.timestamp.astype("string") >= start) & (raw.timestamp.astype("string") < end) & oof.__fold__.ge(0))
    return float(average_precision_score(raw.loc[mask, "fraud"], oof.loc[mask, "pred_1"]))


def main():
    root = Path(ROOT); raw = read_table(root / "train.csv", id_columns=["transaction_id"])
    reports = {key: json.loads((root / f"outputs/reports/{key}.json").read_text()) for key in ("R017", "R023")}
    oofs = {key: pd.read_csv(root / reports[key]["oof_path"], dtype={"transaction_id": "string"}) for key in reports}
    for key, oof in oofs.items():
        if not oof.transaction_id.astype(str).equals(raw.transaction_id.astype(str)): raise ValueError(f"{key} OOF row/transaction-ID alignment failed")
    windows = {"F1": ("2026-03-14", "2026-05-15"), "F2": ("2026-05-15", "2026-07-16"), "F2_early": ("2026-05-15", "2026-06-15"), "F2_late_corrected": ("2026-06-15", "2026-07-16"), "July": ("2026-07-01", "2026-07-16"), "June_15_30": ("2026-06-15", "2026-07-01"), "July_1_7": ("2026-07-01", "2026-07-08"), "July_8_15": ("2026-07-08", "2026-07-16")}
    scores = {key: {name: _score(raw, oofs[key], *window) for name, window in windows.items()} for key in reports}
    delta = {name: scores["R023"][name] - scores["R017"][name] for name in windows}
    # Fixed hierarchy: current evidence is near-flat but negative on July and corrected late.
    decision = "LOSE" if delta["July"] < -0.002 or delta["F2_late_corrected"] < -0.002 else "WIN" if (delta["July"] >= .003 and delta["F2_late_corrected"] >= .002 and delta["F2"] >= .003 and delta["F1"] >= -.002 and delta["July_1_7"] >= -.002 and delta["July_8_15"] >= -.002) else "FLAT-MIXED"
    frame, _ = load_training(load_config(root / "config_r023.json"), root)
    f2 = oofs["R023"].__fold__.eq(1); values = frame.loc[f2, "customer_new_merchants_7d"]
    bins = {"zero": values.eq(0), "one": values.eq(1), "two_to_three": values.between(2, 3), "four_plus": values.ge(4)}
    support = {}
    for name, mask in bins.items():
        labels, preds = raw.loc[f2 & mask, "fraud"], oofs["R023"].loc[f2 & mask, "pred_1"]
        support[name] = {"rows": int(mask.sum()), "positives": int(labels.sum()), "fraud_prevalence": float(labels.mean()), "average_precision": float(average_precision_score(labels, preds)) if labels.nunique() == 2 else None}
    new_importance = {fold: [item for item in reports["R023"]["feature_importances"][fold] if item["feature"].startswith("customer_new_") or item["feature"] == "customer_seconds_since_last_new_merchant"] for fold in ("F2", "F1")}
    gates = {"July_at_least_plus_0.003": delta["July"] >= .003, "F2_late_at_least_plus_0.002": delta["F2_late_corrected"] >= .002, "F2_at_least_plus_0.003": delta["F2"] >= .003, "no_meaningful_F1_degradation": delta["F1"] >= -.002, "neither_July_subperiod_materially_worse": delta["July_1_7"] >= -.002 and delta["July_8_15"] >= -.002}
    report = {"experiment_id": "R023", "parent": "R017", "scores": scores["R023"], "deltas_vs_R017": delta, "decision": decision, "decision_hierarchy": ["July", "corrected F2 late", "F2", "F1", "both July subperiods"], "operational_gates": gates, "operational_gates_cleared": all(gates.values()), "customer_new_merchants_7d_support_F2": support, "new_feature_importances": new_importance, "oof_row_transaction_id_alignment": True, "test_inference": False, "submission_created": False}
    path = root / "outputs/reports/R023/r023_diagnostics_report.json"; path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
