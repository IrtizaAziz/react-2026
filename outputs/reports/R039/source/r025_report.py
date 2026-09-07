"""Post-fit R025 decision and merchant new-customer composition diagnostics."""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from .config import ROOT, load_config
from .data import load_training, read_table


WINDOWS = {"F1": ("2026-03-14", "2026-05-15"), "F2": ("2026-05-15", "2026-07-16"),
           "F2_early": ("2026-05-15", "2026-06-15"), "F2_late_corrected": ("2026-06-15", "2026-07-16"),
           "July": ("2026-07-01", "2026-07-16"), "June_15_30": ("2026-06-15", "2026-07-01"),
           "July_1_7": ("2026-07-01", "2026-07-08"), "July_8_15": ("2026-07-08", "2026-07-16")}


def _ap(labels, predictions): return float(average_precision_score(labels, predictions))


def main():
    root = Path(ROOT); records = {name: json.loads((root / f"outputs/reports/{name}.json").read_text()) for name in ("R017", "R025")}
    raw = read_table(root / "train.csv", id_columns=["transaction_id"])
    oofs = {name: pd.read_csv(root / record["oof_path"], dtype={"transaction_id": "string"}) for name, record in records.items()}
    for name, oof in oofs.items():
        if not oof.transaction_id.astype(str).equals(raw.transaction_id.astype(str)) or not oof.__row__.eq(pd.RangeIndex(len(raw))).all():
            raise ValueError(f"{name} OOF row / transaction-ID alignment failed")
    times = raw.timestamp.astype("string"); scores = {name: {} for name in records}
    for name, oof in oofs.items():
        for window, (start, end) in WINDOWS.items():
            mask = times.ge(start) & times.lt(end) & oof.__fold__.ge(0)
            scores[name][window] = _ap(raw.loc[mask, "fraud"], oof.loc[mask, "pred_1"])
    deltas = {name: scores["R025"][name] - scores["R017"][name] for name in WINDOWS}
    gates = {"July >= +0.003": deltas["July"] >= .003, "corrected F2 late >= +0.002": deltas["F2_late_corrected"] >= .002,
             "F2 >= +0.003": deltas["F2"] >= .003, "no meaningful F1 degradation": deltas["F1"] >= -.002,
             "neither July subperiod materially worse": deltas["July_1_7"] >= -.002 and deltas["July_8_15"] >= -.002}
    decision = "LOSE" if deltas["July"] < -.002 or deltas["F2_late_corrected"] < -.002 else "WIN" if all(gates.values()) else "FLAT-MIXED"
    frame, _ = load_training(load_config(root / "config_r025.json"), root); f2 = oofs["R025"].__fold__.eq(1)
    count = frame.merchant_new_customers_24h
    buckets = {"no recent new customers": count.eq(0), "low influx (1)": count.eq(1), "moderate influx (2-4)": count.between(2, 4), "high influx (5+)": count.ge(5)}
    support = {}
    for name, bucket in buckets.items():
        mask = f2 & bucket; labels, predictions = raw.loc[mask, "fraud"], oofs["R025"].loc[mask, "pred_1"]
        support[name] = {"rows": int(mask.sum()), "positives": int(labels.sum()), "fraud_prevalence": float(labels.mean()), "average_precision": _ap(labels, predictions) if labels.nunique() == 2 else None}
    new = ["merchant_new_customers_24h", "merchant_new_customers_7d", "merchant_new_customer_share_24h", "merchant_new_customer_share_7d", "merchant_seconds_since_last_new_customer"]
    existing = ["merchant_prior_transaction_count", "merchant_prior_unique_customers", "merchant_transactions_24h", "merchant_unique_customers_24h"]
    correlations = frame.loc[f2, new + existing].corr(method="spearman").loc[new, existing].to_dict()
    importances = {fold: [item for item in records["R025"]["feature_importances"][fold] if item["feature"] in new] for fold in ("F2", "F1")}
    report = {"experiment_id": "R025", "parent": "R017", "scores": scores["R025"], "deltas_vs_R017": deltas,
              "decision": decision, "decision_hierarchy": ["July", "corrected F2 late", "F2", "F1", "both July subperiods"],
              "scarce_submission_gates": gates, "scarce_submission_gates_cleared": all(gates.values()),
              "merchant_new_customer_influx_buckets_F2": support, "spearman_correlations_F2_new_features_vs_R017_merchant_scale": correlations,
              "new_feature_importances_descriptive_only": importances, "oof_row_transaction_id_alignment": True,
              "test_inference": False, "submission_created": False}
    path = root / "outputs/reports/R025/r025_diagnostics_report.json"; path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
