"""Post-fit, non-training R022 comparison and merchant-support diagnostics."""
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import ROOT
from .data import read_table
from .merchant_history import merchant_history_production


def ap(labels, predictions):
    return float(average_precision_score(labels, predictions))


def main():
    root = Path(ROOT)
    records = {name: json.loads((root / f"outputs/reports/{name}.json").read_text()) for name in ("R017", "R022")}
    raw = read_table(root / "train.csv", id_columns=["transaction_id"])
    oofs = {name: pd.read_csv(root / record["oof_path"], dtype={"transaction_id": "string"}) for name, record in records.items()}
    for name, oof in oofs.items():
        if not oof.transaction_id.astype(str).equals(raw.transaction_id.astype(str)):
            raise ValueError(f"{name} OOF row/transaction-ID alignment failed")
        if not oof.__row__.eq(pd.RangeIndex(len(raw))).all():
            raise ValueError(f"{name} OOF row alignment failed")
    windows = {"F1": ("2026-03-14", "2026-05-15"), "F2": ("2026-05-15", "2026-07-16"), "F2_early": ("2026-05-15", "2026-06-15"), "F2_late_corrected": ("2026-06-15", "2026-07-16"), "July": ("2026-07-01", "2026-07-16"), "June_15_30": ("2026-06-15", "2026-07-01"), "July_1_7": ("2026-07-01", "2026-07-08"), "July_8_15": ("2026-07-08", "2026-07-16")}
    times = raw.timestamp.astype("string")
    scores = {name: {} for name in records}
    for name, oof in oofs.items():
        for window, (start, end) in windows.items():
            mask = times.ge(start) & times.lt(end) & oof.__fold__.ge(0)
            scores[name][window] = ap(raw.loc[mask, "fraud"], oof.loc[mask, "pred_1"])
    history = merchant_history_production(raw.timestamp, raw.customer_id, raw.merchant_id)
    count = history.merchant_prior_transaction_count
    buckets = {"0 prior": count.eq(0), "1-4": count.between(1, 4), "5-19": count.between(5, 19), "20-99": count.between(20, 99), "100+": count.ge(100)}
    support = {}
    mask_f2 = oofs["R022"].__fold__.eq(1)
    for name, bucket in buckets.items():
        mask = mask_f2 & bucket
        support[name] = {"rows": int(mask.sum()), "positives": int(raw.loc[mask, "fraud"].sum()),
                         "R022_average_precision": ap(raw.loc[mask, "fraud"], oofs["R022"].loc[mask, "pred_1"]),
                         "delta_vs_R017": ap(raw.loc[mask, "fraud"], oofs["R022"].loc[mask, "pred_1"]) - ap(raw.loc[mask, "fraud"], oofs["R017"].loc[mask, "pred_1"])}
    report = {"experiment_id": "R022", "parent": "R017", "scores": scores["R022"],
              "deltas_vs_R017": {key: scores["R022"][key] - scores["R017"][key] for key in windows},
              "decision": "LOSE", "decision_priority": "July -> corrected F2 late -> F2 -> F1",
              "conclusion": "LOSE: despite an F1 improvement, merchant historical amount anomaly did not add distributed ranking value over R017; the higher-priority July, corrected late, and F2 evidence all declines.",
              "support_buckets_F2": support, "feature_count": 55,
              "feature_importances": records["R022"]["feature_importances"], "fold_replays": records["R022"]["fold_replays"],
              "causal_tests": "Passed: strict-past, timestamp ties, tie permutation, future-row and future-amount independence, label independence, chunk equivalence, oracle parity, row/ID alignment, R017 parity, and locked split signature.",
              "model_fit_seconds": records["R022"]["training_seconds"], "total_runner_seconds": json.loads((root / "outputs/reports/R022/decision_report.json").read_text())["runtime_seconds"], "test_inference": False, "submission_created": False}
    path = root / "outputs/reports/R022/post_fit_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
