"""R039: infrastructure-corrected, fixed R029/R037 OOF-only blend."""
from pathlib import Path
import time
import numpy as np
import pandas as pd

from .config import Config, ROOT
from .data import read_table
from .metrics import score
from .predict import read_predictions, save_predictions
from .utils import environment, finish_record, read_json, reserve_experiment, save_json, snapshot_source, sha256

MEMBERS = ("R029", "R037")
WEIGHTS = (0.75, 0.25)
WINDOWS = {
    "F2_early": ("2026-05-15", "2026-06-15"),
    "F2_late_corrected": ("2026-06-15", "2026-07-16"),
    "June_15_30": ("2026-06-15", "2026-07-01"),
    "July_1_15": ("2026-07-01", "2026-07-16"),
    "July_1_7": ("2026-07-01", "2026-07-08"),
    "July_8_15": ("2026-07-08", "2026-07-16"),
}


def blend_covered_probabilities(a, b, weights=WEIGHTS):
    """Fail closed unless null coverage is identical; preserve nulls outside it."""
    weights = tuple(weights)
    if weights != WEIGHTS:
        raise ValueError("R039 weights are locked at exactly 0.75/0.25")
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("R039 requires aligned binary probability matrices")
    mask_a, mask_b = ~np.isnan(a).any(axis=1), ~np.isnan(b).any(axis=1)
    if not np.array_equal(mask_a, mask_b):
        raise ValueError("Member OOF coverage masks differ; refusing to blend")
    for item in (a[mask_a], b[mask_b]):
        if not np.isfinite(item).all() or (item < 0).any() or (item > 1).any() or not np.allclose(item.sum(axis=1), 1, atol=1e-10, rtol=0):
            raise ValueError("Non-null member output is not a valid finite probability")
    out = np.full(a.shape, np.nan, dtype=float)
    out[mask_a] = weights[0] * a[mask_a] + weights[1] * b[mask_a]
    return out, mask_a


def _metric_map(record):
    return {"F1": record["fold_scores"][0], "F2": record["fold_scores"][1],
            "F2_early": record["diagnostics"].get("F2_early", record["diagnostics"].get("F2_early_half"))["average_precision"],
            "F2_late_corrected": record["diagnostics"].get("F2_late_corrected", record["diagnostics"].get("F2_late_half"))["average_precision"],
            **{key: record["diagnostics"][key]["average_precision"] for key in ("June_15_30", "July_1_15", "July_1_7", "July_8_15")}}


def run(root=ROOT, experiment="R039"):
    root = Path(root); start = time.perf_counter()
    if experiment != "R039": raise ValueError("This locked retry may only create R039")
    records = {member: read_json(root / f"outputs/reports/{member}.json") for member in MEMBERS}
    if any(item.get("status") != "completed" for item in records.values()): raise ValueError("All immutable members must be completed")
    items = {member: read_predictions(root / records[member]["oof_path"]) for member in MEMBERS}
    a_frame, a_values, a_meta = items["R029"]; b_frame, b_values, b_meta = items["R037"]
    compatibility = ("identity_hash", "data_fingerprint", "class_order", "prediction_kind", "prediction_columns", "id_columns", "split_signature")
    if any(a_meta[key] != b_meta[key] for key in compatibility): raise ValueError("Member metadata mismatch")
    if not a_frame[["__row__", "transaction_id", "__fold__"]].equals(b_frame[["__row__", "transaction_id", "__fold__"]]): raise ValueError("Transaction IDs/order/folds mismatch")
    if a_frame.transaction_id.duplicated().any() or not np.array_equal(a_frame.__row__.to_numpy(), np.arange(len(a_frame))): raise ValueError("Invalid transaction-ID universe/order")
    if records["R029"]["split_signature"] != a_meta["split_signature"] or records["R037"]["split_signature"] != a_meta["split_signature"]: raise ValueError("Record split signature mismatch")
    if records["R029"]["config"]["target"] != records["R037"]["config"]["target"]: raise ValueError("Target mismatch")
    blended, covered = blend_covered_probabilities(a_values, b_values)
    assignment = a_frame["__fold__"].to_numpy()
    expected = assignment >= 0
    if not np.array_equal(covered, expected): raise ValueError("Common OOF mask differs from locked temporal-fold validation population")
    config = Config(**records["R029"]["config"]); config.model = "ensemble"; config.model_params = {"members": list(MEMBERS), "weights": list(WEIGHTS), "method": "weighted"}; config.predict_test = False; config.aggregation = None
    hypothesis = "Exact infrastructure-corrected retry of the failed R038 fixed blend; no scientific hypothesis change."
    change = "0.75*R029 OOF probability + 0.25*R037 OOF probability only on their identical locked OOF coverage; expected warmup nulls remain null. No training, inference, refit, or submission."
    record = reserve_experiment(root, experiment, config, hypothesis, change, parent="R029")
    try:
        provenance = root / record["provenance_path"]; record["source_hashes"] = snapshot_source(provenance / "source")
        save_json(provenance / "blend_spec.json", {"experiment_id": experiment, "members": list(MEMBERS), "weights": list(WEIGHTS), "method": "weighted_probability", "test_predictions": "NOT GENERATED", "submission": "NOT GENERATED"}, exclusive=True)
        save_json(provenance / "environment.json", environment(), exclusive=True)
        save_json(provenance / "member_identity_hash_report.json", {m: {"oof_path": records[m]["oof_path"], "oof_sha256": sha256(root / records[m]["oof_path"]), "metadata": items[m][2]} for m in MEMBERS}, exclusive=True)
        save_json(provenance / "oof_coverage_null_mask_report.json", {"mask_identity": True, "covered_rows": int(covered.sum()), "uncovered_rows": int((~covered).sum()), "expected_locked_temporal_fold_mask": True, "uncovered_predictions_preserved_null": True}, exclusive=True)
        save_json(provenance / "splits.json", read_json(root / records["R029"]["splits_path"]), exclusive=True)
        raw = read_table(root / "train.csv", id_columns=["transaction_id"])
        if not a_frame.transaction_id.equals(raw.transaction_id.astype("string")): raise ValueError("OOF/training transaction-ID order mismatch")
        fold_scores = [score(raw.loc[assignment == fold, "fraud"], blended[assignment == fold], config) for fold in range(2)]
        timestamp = pd.to_datetime(raw.timestamp, errors="raise"); diagnostics = {}
        for name, (begin, end) in WINDOWS.items():
            mask = (assignment == 1) & (timestamp >= begin) & (timestamp < end)
            diagnostics[name] = {"fold": 1, "rows": int(mask.sum()), "positives": int(raw.loc[mask, "fraud"].sum()), "average_precision": score(raw.loc[mask, "fraud"], blended[mask], config)}
        metrics = {"F1": fold_scores[0], "F2": fold_scores[1], **{key: value["average_precision"] for key, value in diagnostics.items()}}
        bases = {member: _metric_map(records[member]) for member in ("R029", "R037", "R017")}
        deltas = {member: {key: float(metrics[key] - bases[member][key]) for key in metrics} for member in bases}
        floors = {"F2": metrics["F2"] >= bases["R017"]["F2"] + .003, "corrected_late": metrics["F2_late_corrected"] >= bases["R017"]["F2_late_corrected"] + .002, "July": metrics["July_1_15"] >= bases["R017"]["July_1_15"] + .003, "F1": metrics["F1"] >= bases["R017"]["F1"] - .002, "July_1_7": metrics["July_1_7"] >= bases["R017"]["July_1_7"] - .002, "July_8_15": metrics["July_8_15"] >= bases["R017"]["July_8_15"] - .002, "integrity": True}
        passed = all(floors.values()); classification = "CLEAR WIN" if passed else "FLAT-NO-SUBMISSION"
        oof_path = f"outputs/oof/{experiment}.csv"; metadata = save_predictions(root / oof_path, a_frame, blended, config, a_meta["data_fingerprint"], folds=assignment, split_signature=a_meta["split_signature"])
        report = {"experiment_id": experiment, "metrics": metrics, "deltas": deltas, "submission_floors": floors, "submission_gate_pass": passed, "classification": classification, "special_diagnostics": {"July_8_15_improves_vs_R037": deltas["R037"]["July_8_15"] > 0, "F2_improves_vs_R029": deltas["R029"]["F2"] > 0, "July_improves_vs_R029": deltas["R029"]["July_1_15"] > 0}, "test_predictions": "NOT GENERATED", "submission": "NOT GENERATED"}
        save_json(provenance / "decision_report.json", report, exclusive=True)
        (provenance / "experiment_card.md").write_text(f"# R039 experiment card\n\nFixed OOF-only blend: `0.75 * R029 + 0.25 * R037`. Classification: **{classification}**.\n\nExpected expanding-window warmup predictions are intentionally null; no model/test/submission artifact exists.\n", encoding="utf-8", newline="\n")
        save_json(provenance / "provenance.json", {"R038_preserved": True, "members": list(MEMBERS), "member_oof_hashes": {m: sha256(root / records[m]["oof_path"]) for m in MEMBERS}, "r039_oof_metadata": metadata}, exclusive=True)
        record.update(ensemble=config.model_params, train_fingerprint=a_meta["data_fingerprint"], split_signature=a_meta["split_signature"], splits_path=f"{record['provenance_path']}splits.json", oof_path=oof_path, fold_scores=fold_scores, cv_mean=float(np.mean(fold_scores)), cv_std=float(np.std(fold_scores)), diagnostics=diagnostics, oof_coverage=float(covered.mean()), pooled_covered_oof_score=score(raw.loc[covered, "fraud"], blended[covered], config), decision=classification, decision_report_path=f"{record['provenance_path']}decision_report.json", training_seconds=time.perf_counter()-start, status="completed", test_inference=False, submission_created=False, pre_fit_assertions={"member_identity_and_split_verified": True, "identical_expected_null_masks_accepted": True, "expected_nulls_preserved": True, "fixed_weights": list(WEIGHTS), "no_training": True, "no_test_prediction": True})
        finish_record(root, record)
        from .experiment_docs import rebuild_experiment_index
        rebuild_experiment_index(root)
        return report
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", training_seconds=time.perf_counter()-start); finish_record(root, record); raise


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
