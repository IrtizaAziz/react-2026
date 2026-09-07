"""R044: locked OOF-only blend of R040 and R041; no inference or submission."""
from pathlib import Path
import json
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .config import Config, ROOT
from .ensemble import blend_covered_oofs, load_oof_scoring_frame, score_canonical_oof
from .predict import read_predictions, save_predictions
from .utils import environment, finish_record, read_json, reserve_experiment, save_json, sha256, snapshot_source

EXPERIMENT = "R044"
MEMBERS = ("R040", "R041")
WEIGHTS = (0.75, 0.25)
COMPARATORS = ("R040", "R041", "R029")
WINDOWS = (("F2_early", 1, "2026-05-15", "2026-06-15"),
           ("F2_late_corrected", 1, "2026-06-15", "2026-07-16"),
           ("June_15_30", 1, "2026-06-15", "2026-07-01"),
           ("July", 1, "2026-07-01", "2026-07-16"),
           ("July_1_7", 1, "2026-07-01", "2026-07-08"),
           ("July_8_15", 1, "2026-07-08", "2026-07-16"))


def _top_overlap(a, b, size):
    return int(len(set(np.argpartition(a, -size)[-size:]) & set(np.argpartition(b, -size)[-size:])))


def _diversity(a, b, mask):
    a, b = a[mask, 1], b[mask, 1]
    return {"rows": int(mask.sum()), "pearson": float(pearsonr(a, b).statistic),
            "spearman": float(spearmanr(a, b).statistic),
            "top_1000_overlap": _top_overlap(a, b, 1000), "top_5000_overlap": _top_overlap(a, b, 5000)}


def run(root=ROOT, experiment=EXPERIMENT):
    root = Path(root)
    if experiment != EXPERIMENT or MEMBERS != ("R040", "R041") or WEIGHTS != (0.75, 0.25):
        raise ValueError("R044 is locked to exactly 0.75 * R040 + 0.25 * R041")
    start = time.perf_counter()
    records = {name: read_json(root / f"outputs/reports/{name}.json") for name in (*COMPARATORS,)}
    if any(record.get("status") != "completed" for record in records.values()):
        raise ValueError("All R044 sources must be completed")
    items = {name: read_predictions(root / records[name]["oof_path"]) for name in COMPARATORS}
    a_frame, a_values, a_meta = items["R040"]
    b_frame, b_values, b_meta = items["R041"]
    fields = ("identity_hash", "data_fingerprint", "class_order", "prediction_kind", "prediction_columns", "id_columns", "split_signature")
    if any(a_meta[key] != b_meta[key] for key in fields):
        raise ValueError("R040/R041 metadata provenance mismatch")
    identity_columns = ["__row__", "transaction_id", "__fold__"]
    if not a_frame[identity_columns].equals(b_frame[identity_columns]):
        raise ValueError("R040/R041 transaction IDs, row order, or fold assignments differ")
    if a_frame.transaction_id.duplicated().any() or not np.array_equal(a_frame.__row__.to_numpy(), np.arange(len(a_frame))):
        raise ValueError("Invalid canonical OOF transaction universe")
    for name in COMPARATORS:
        if records[name]["split_signature"] != a_meta["split_signature"]:
            raise ValueError(f"{name} record split signature mismatch")
    config = Config(**records["R040"]["config"])
    raw, fingerprint = load_oof_scoring_frame(root, config)
    if fingerprint != a_meta["data_fingerprint"] or not a_frame.transaction_id.equals(raw.transaction_id.astype("string")):
        raise ValueError("OOF transaction IDs or labels/training fingerprint do not match canonical data")
    _frame, blended, _metadata, covered = blend_covered_oofs([items[name] for name in MEMBERS], WEIGHTS)
    expected = a_frame["__fold__"].to_numpy() >= 0
    if not np.array_equal(covered, expected):
        raise ValueError("Jointly scored mask differs from canonical validation mask")
    for name in COMPARATORS:
        values = items[name][1]
        if not np.isfinite(values[covered]).all() or (values[covered] < 0).any() or (values[covered] > 1).any():
            raise ValueError(f"{name} has invalid scored probabilities")
    assignment = a_frame["__fold__"].to_numpy()
    metric_sets = {name: score_canonical_oof(raw, items[name][1], assignment, config, WINDOWS) for name in COMPARATORS}
    metrics = score_canonical_oof(raw, blended, assignment, config, WINDOWS)
    deltas = {name: {key: float(metrics[key] - values[key]) for key in metrics} for name, values in metric_sets.items()}
    timestamps = __import__("pandas").to_datetime(raw[config.time_column], errors="raise")
    f2 = assignment == 1
    july = f2 & (timestamps >= "2026-07-01") & (timestamps < "2026-07-16")
    diversity = {"F2": _diversity(a_values, b_values, f2), "July": _diversity(a_values, b_values, july)}
    gates = {"F2": metrics["F2"] > metric_sets["R040"]["F2"],
             "corrected_late": metrics["F2_late_corrected"] >= metric_sets["R040"]["F2_late_corrected"],
             "July": metrics["July"] >= metric_sets["R040"]["July"],
             "July_1_7": metrics["July_1_7"] >= metric_sets["R040"]["July_1_7"] - .0005,
             "July_8_15": metrics["July_8_15"] >= metric_sets["R040"]["July_8_15"],
             "F1": metrics["F1"] >= metric_sets["R040"]["F1"] - .001,
             "integrity_replay": True}
    submission_worthy = all(gates.values())
    clear_win = submission_worthy and metrics["F2"] >= metric_sets["R040"]["F2"] + .0005 and metrics["F2_late_corrected"] > metric_sets["R040"]["F2_late_corrected"] and metrics["July"] > metric_sets["R040"]["July"] and metrics["July_1_7"] >= metric_sets["R040"]["July_1_7"] and metrics["July_8_15"] >= metric_sets["R040"]["July_8_15"]
    classification = "CLEAR-WIN" if clear_win else ("SUBMISSION-WORTHY" if submission_worthy else "NOT-SUBMISSION-WORTHY")
    record = read_json(root / f"outputs/reports/{experiment}.json")
    if record.get("status") != "completed":
        raise ValueError("R044's supervised immutable OOF blend must complete before evaluation")
    r044_frame, r044_values, r044_meta = read_predictions(root / record["oof_path"])
    if not r044_frame["transaction_id"].equals(a_frame["transaction_id"]) or not np.allclose(r044_values[covered], blended[covered], atol=0, rtol=0) or not np.isnan(r044_values[~covered]).all():
        raise ValueError("R044 OOF does not exactly replay the locked fixed blend")
    source_hashes = {name: {"oof_path": records[name]["oof_path"], "oof_sha256": sha256(root / records[name]["oof_path"]), "metadata_sha256": sha256((root / records[name]["oof_path"]).with_suffix(".meta.json")), "metadata": items[name][2]} for name in MEMBERS}
    integrity = {"identical_oof_transaction_ids_order_and_rows": True, "identical_labels": True, "identical_fold_assignments": True, "identical_split_signature": True, "identical_scored_null_coverage_mask": True, "canonical_scored_rows": int(covered.sum()), "canonical_null_rows": int((~covered).sum()), "probabilities_finite_and_in_unit_interval": True, "source_artifact_hashes_and_provenance_verified": True, "exact_r044_oof_replay": True}
    report = {"experiment_id": experiment, "formula": "0.75 * R040_probability + 0.25 * R041_probability", "integrity": integrity, "metrics": metrics, "deltas": deltas, "diversity_R040_vs_R041": diversity, "submission_worthy": {"pass": submission_worthy, "gates": gates}, "clear_win": {"pass": clear_win}, "classification": classification, "test_predictions": "NOT GENERATED", "submission": "NOT GENERATED", "oof_metadata": r044_meta, "source_artifact_hashes": source_hashes, "runtime_seconds": time.perf_counter() - start}
    save_json(root / record["provenance_path"] / "r044_evaluation_report.json", report, exclusive=True)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
