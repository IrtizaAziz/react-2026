"""Post-fit, non-training diagnostics for the immutable R012 comparison."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import ROOT
from .customer_amount_30d import customer_amount_30d_production
from .customer_history import customer_history_production
from .data import read_table


def main():
    root = Path(ROOT)
    r012 = json.loads((root / "outputs/reports/R012.json").read_text(encoding="utf-8"))
    r007 = json.loads((root / "outputs/reports/R007.json").read_text(encoding="utf-8"))
    raw = read_table(root / "train.csv", id_columns=["transaction_id"])
    lifetime = customer_history_production(raw["timestamp"], raw["customer_id"], raw["amount_bdt"])
    rolling = customer_amount_30d_production(raw["timestamp"], raw["customer_id"], raw["amount_bdt"],
                                             lifetime["customer_prior_mean_amount"])
    oof = pd.read_csv(root / r012["oof_path"], dtype={"transaction_id": "string"})
    f2 = oof["__fold__"].eq(1)
    if not f2.any() or not np.array_equal(oof.loc[f2, "transaction_id"].astype(str).to_numpy(), raw.loc[f2, "transaction_id"].astype(str).to_numpy()):
        # OOF rows preserve source order; this assertion makes the positional support join explicit.
        raise ValueError("R012 F2 OOF alignment is not the raw training order")
    support = rolling["customer_prior_30d_count"].to_numpy()[f2.to_numpy()]
    labels = raw.loc[f2, "fraud"].to_numpy()
    predictions = oof.loc[f2, "pred_1"].to_numpy()
    groups = {"0": support == 0, "1-4": (support >= 1) & (support <= 4), ">=5": support >= 5}
    diagnostics = {}
    for name, mask in groups.items():
        diagnostics[name] = {"rows": int(mask.sum()), "positives": int(labels[mask].sum()),
                             "prevalence": float(labels[mask].mean()),
                             "average_precision": float(average_precision_score(labels[mask], predictions[mask]))}
    names = {"F1": ("fold_scores", 0), "F2": ("fold_scores", 1), "mean": ("cv_mean", None),
             "early": ("diagnostics", "F2_early_half"), "late": ("diagnostics", "F2_late_half"),
             "July": ("diagnostics", "July_1_15"), "June_15_30": ("diagnostics", "June_15_30"),
             "July_1_7": ("diagnostics", "July_1_7"), "July_8_15": ("diagnostics", "July_8_15")}
    def metric(record, source, key):
        return record[source][key] if source == "fold_scores" else record[source][key]["average_precision"] if source == "diagnostics" else record[source]
    deltas = {name: float(metric(r012, source, key) - metric(r007, source, key)) for name, (source, key) in names.items()}
    report = {"experiment_id": "R012", "parent": "R007", "causal_parity_preflight": "passed",
              "support_diagnostics_f2": diagnostics, "deltas_vs_r007": deltas,
              "decision": "FLAT-MIXED", "next_predefined_branch": "customer-merchant familiarity from R007",
              "comparison_basis": "identical saved validation rows; original R007 OOF artifact remains unmodified"}
    path = root / "outputs/reports/R012/decision_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
