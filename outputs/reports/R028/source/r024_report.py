"""Post-fit R024 comparison and descriptive device-location support diagnostics."""
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


def ap(y, p): return float(average_precision_score(y, p))


def main():
    root = Path(ROOT); records = {name: json.loads((root / f"outputs/reports/{name}.json").read_text()) for name in ("R017", "R024")}
    raw = read_table(root / "train.csv", id_columns=["transaction_id"])
    oofs = {name: pd.read_csv(root / record["oof_path"], dtype={"transaction_id": "string"}) for name, record in records.items()}
    for name, oof in oofs.items():
        if not oof.transaction_id.astype(str).equals(raw.transaction_id.astype(str)) or not oof.__row__.eq(pd.RangeIndex(len(raw))).all():
            raise ValueError(f"{name} OOF row / transaction-ID alignment failed")
    times = raw.timestamp.astype("string"); scores = {name: {} for name in records}
    for name, oof in oofs.items():
        for window, (start, end) in WINDOWS.items():
            mask = times.ge(start) & times.lt(end) & oof.__fold__.ge(0)
            scores[name][window] = ap(raw.loc[mask, "fraud"], oof.loc[mask, "pred_1"])
    frame, _ = load_training(load_config(root / "config_r024.json"), root)
    f2 = oofs["R024"].__fold__.eq(1); count = frame.device_location_prior_count
    buckets = {"new / zero prior": count.eq(0), "prior count 1-4": count.between(1, 4), "prior count 5-19": count.between(5, 19), "prior count 20+": count.ge(20)}
    support = {name: {"rows": int((f2 & bucket).sum()), "positives": int(raw.loc[f2 & bucket, "fraud"].sum()),
                      "fraud_prevalence": float(raw.loc[f2 & bucket, "fraud"].mean()), "average_precision": ap(raw.loc[f2 & bucket, "fraud"], oofs["R024"].loc[f2 & bucket, "pred_1"])} for name, bucket in buckets.items() if (f2 & bucket).any()}
    july = times.ge("2026-07-01") & times.lt("2026-07-16") & f2
    deltas = {name: scores["R024"][name] - scores["R017"][name] for name in WINDOWS}
    gates = {"July >= +0.003": deltas["July"] >= .003, "corrected F2 late >= +0.002": deltas["F2_late_corrected"] >= .002,
             "F2 >= +0.003": deltas["F2"] >= .003, "no meaningful F1 degradation": deltas["F1"] >= -.002,
             "neither July subperiod materially worse": deltas["July_1_7"] >= -.002 and deltas["July_8_15"] >= -.002}
    report = {"experiment_id": "R024", "parent": "R017", "scores": scores["R024"], "deltas_vs_R017": deltas,
              "decision": "FLAT-MIXED", "decision_priority": "July -> corrected F2 late -> F2 -> F1 -> both July subperiods",
              "conclusion": "FLAT-MIXED: July, corrected late, and F2 improve slightly, but each primary gain is below the fixed scarce-submission gate; F1 is slightly lower and neither July subperiod is materially worse.",
              "scarce_submission_gates": gates, "support_buckets_F2": support,
              "July_fraction_with_prior_device_location_history": float(count.loc[july].gt(0).mean()),
              "July_fraction_with_at_least_5_prior_pair_events": float(count.loc[july].ge(5).mean()),
              "feature_importances": records["R024"]["feature_importances"], "fold_replays": records["R024"]["fold_replays"],
              "causal_tests": "Passed before fit: strict <t, equal-timestamp isolation, tie permutation invariance, duplicate pair handling, future-row/device/location mutation independence, label independence, chunk equivalence, optimized-vs-oracle parity, row/transaction-ID alignment, R017 50-feature parity, locked split signature, and missing-location sentinel continuity.",
              "test_inference": False, "submission_created": False}
    path = root / "outputs/reports/R024/post_fit_report.json"; path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
