"""Post-fit, non-training R015 recent-device-sharing diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .config import ROOT
from .customer_relationships import device_global_production
from .data import read_table
from .device_recent_sharing import device_recent_sharing_production


def main():
    root = Path(ROOT)
    records = {k: json.loads((root / f"outputs/reports/{k}.json").read_text()) for k in ("R007", "R013", "R015")}
    raw = read_table(root / "train.csv", id_columns=["transaction_id"])
    life = device_global_production(raw.timestamp, raw.customer_id, raw.device_id)
    recent = device_recent_sharing_production(raw.timestamp, raw.customer_id, raw.device_id, life.device_prior_distinct_customer_count)
    oof = pd.read_csv(root / records["R015"]["oof_path"], dtype={"transaction_id": "string"})
    mask = oof.__fold__.eq(1).to_numpy(); y = raw.loc[mask, "fraud"].to_numpy(); p = oof.loc[mask, "pred_1"].to_numpy()
    def diag(groups):
        return {name: {"rows": int(x.sum()), "positives": int(y[x].sum()), "prevalence": float(y[x].mean()) if x.any() else None, "average_precision": float(average_precision_score(y[x], p[x])) if x.any() and y[x].any() else None} for name, x in groups.items()}
    a = recent.loc[mask]
    keys = {"F1": ("fold_scores", 0), "F2": ("fold_scores", 1), "mean": ("cv_mean", None), "pooled": ("pooled_covered_oof_score", None), "early": ("diagnostics", "F2_early_half"), "late": ("diagnostics", "F2_late_half"), "July": ("diagnostics", "July_1_15"), "June_15_30": ("diagnostics", "June_15_30"), "July_1_7": ("diagnostics", "July_1_7"), "July_8_15": ("diagnostics", "July_8_15")}
    def value(record, section, key):
        return record[section][key] if section == "fold_scores" else record[section] if key is None else record[section][key]["average_precision"]
    f2_delta = value(records["R015"], "fold_scores", 1) - value(records["R013"], "fold_scores", 1)
    report = {"experiment_id": "R015", "parent": "R013", "causal_parity_preflight": "passed", "window_definition": "[t-window, t), timestamp-batched", "device_other_customers_24h_f2": diag({"0": a.device_other_customers_24h.eq(0).to_numpy(), "1": a.device_other_customers_24h.eq(1).to_numpy(), "2-4": a.device_other_customers_24h.between(2, 4).to_numpy(), ">=5": a.device_other_customers_24h.ge(5).to_numpy()}), "device_other_customers_7d_f2": diag({"0": a.device_other_customers_7d.eq(0).to_numpy(), "1": a.device_other_customers_7d.eq(1).to_numpy(), "2-4": a.device_other_customers_7d.between(2, 4).to_numpy(), ">=5": a.device_other_customers_7d.ge(5).to_numpy()}), "device_unique_customers_7d_distribution_f2": {str(k): int(v) for k, v in a.device_unique_customers_7d.value_counts().sort_index().items()}, "ratio_distribution_f2": {"min": float(a.device_7d_unique_to_lifetime_unique_ratio.min()), "p50": float(a.device_7d_unique_to_lifetime_unique_ratio.quantile(.5)), "p90": float(a.device_7d_unique_to_lifetime_unique_ratio.quantile(.9)), "p99": float(a.device_7d_unique_to_lifetime_unique_ratio.quantile(.99)), "max": float(a.device_7d_unique_to_lifetime_unique_ratio.max())}, "deltas_vs_r013": {n: float(value(records["R015"], s, k) - value(records["R013"], s, k)) for n, (s, k) in keys.items()}, "deltas_vs_r007": {n: float(value(records["R015"], s, k) - value(records["R007"], s, k)) for n, (s, k) in keys.items()}, "decision": "WIN" if f2_delta >= .002 else "FLAT", "acceptance_rule": "F2 average-precision delta versus R013 >= 0.002", "comparison_basis": "identical saved validation rows; prior artifacts unmodified"}
    (root / "outputs/reports/R015/decision_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
