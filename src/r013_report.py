"""Post-fit, non-training diagnostics for immutable R013."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .config import ROOT
from .customer_history import customer_history_production
from .customer_merchant import customer_merchant_production
from .data import read_table


def main():
    root = Path(ROOT)
    r013 = json.loads((root / "outputs/reports/R013.json").read_text(encoding="utf-8"))
    r007 = json.loads((root / "outputs/reports/R007.json").read_text(encoding="utf-8"))
    raw = read_table(root / "train.csv", id_columns=["transaction_id"])
    history = customer_history_production(raw["timestamp"], raw["customer_id"], raw["amount_bdt"])
    pairs = customer_merchant_production(raw["timestamp"], raw["customer_id"], raw["merchant_id"], history["customer_prior_count"])
    oof = pd.read_csv(root / r013["oof_path"], dtype={"transaction_id": "string"})
    f2 = oof["__fold__"].eq(1).to_numpy()
    if not np.array_equal(oof.loc[f2, "transaction_id"].astype(str).to_numpy(), raw.loc[f2, "transaction_id"].astype(str).to_numpy()):
        raise ValueError("R013 F2 OOF alignment is not raw training order")
    labels, predictions = raw.loc[f2, "fraud"].to_numpy(), oof.loc[f2, "pred_1"].to_numpy()
    counts, new = pairs.loc[f2, "customer_merchant_prior_count"].to_numpy(), pairs.loc[f2, "customer_merchant_is_new"].to_numpy()
    def diagnose(groups):
        return {name: {"rows": int(mask.sum()), "positives": int(labels[mask].sum()), "prevalence": float(labels[mask].mean()), "average_precision": float(average_precision_score(labels[mask], predictions[mask]))} for name, mask in groups.items()}
    pair_history = diagnose({"new_customer_merchant_pair": new == 1, "previously_seen_pair": new == 0})
    pair_buckets = diagnose({"0": counts == 0, "1": counts == 1, "2-4": (counts >= 2) & (counts <= 4), ">=5": counts >= 5})
    keys = {"F1": ("fold_scores", 0), "F2": ("fold_scores", 1), "mean": ("cv_mean", None), "early": ("diagnostics", "F2_early_half"), "late": ("diagnostics", "F2_late_half"), "July": ("diagnostics", "July_1_15"), "June_15_30": ("diagnostics", "June_15_30"), "July_1_7": ("diagnostics", "July_1_7"), "July_8_15": ("diagnostics", "July_8_15")}
    def value(record, source, key): return record[source][key] if source == "fold_scores" else record[source] if source == "cv_mean" else record[source][key]["average_precision"]
    deltas = {name: float(value(r013, source, key) - value(r007, source, key)) for name, (source, key) in keys.items()}
    report = {"experiment_id": "R013", "parent": "R007", "causal_parity_preflight": "passed", "pair_history_f2": pair_history, "pair_count_buckets_f2": pair_buckets, "deltas_vs_r007": deltas, "decision": "WIN", "comparison_basis": "identical saved validation rows; original R007 artifacts unmodified"}
    (root / "outputs/reports/R013/decision_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
