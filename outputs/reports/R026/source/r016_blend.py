"""R016: one preregistered, OOF-only R013/R011 arithmetic blend."""
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

from .config import Config, ROOT
from .data import read_table
from .metrics import score
from .predict import save_predictions
from .utils import environment, finish_record, read_json, reserve_experiment, save_json, snapshot_source

MEMBERS = ("R013", "R011")
WEIGHTS = (0.75, 0.25)
SPLIT = "6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47"
WINDOWS = {"F2_early_half": ("2026-05-15", "2026-06-15"), "F2_late_half": ("2026-06-15", "2026-07-16"), "July_1_15": ("2026-07-01", "2026-07-16"), "June_15_30": ("2026-06-15", "2026-07-01"), "July_1_7": ("2026-07-01", "2026-07-08"), "July_8_15": ("2026-07-08", "2026-07-16")}


def _oof(root, experiment):
    record = read_json(root / f"outputs/reports/{experiment}.json")
    frame = pd.read_csv(root / record["oof_path"], dtype={"transaction_id": "string"})
    metadata = read_json(root / f"{record['oof_path'][:-4]}.meta.json")
    actual = hashlib.sha256((root / record["oof_path"]).read_bytes()).hexdigest()
    if record["status"] != "completed" or actual != metadata["sha256"] or metadata["split_signature"] != SPLIT or record["split_signature"] != SPLIT:
        raise ValueError(f"{experiment} OOF provenance/hash/split verification failed")
    return record, frame, metadata, actual


def main():
    root = Path(ROOT); start = time.perf_counter()
    r013, a, meta_a, hash_a = _oof(root, "R013"); r011, b, meta_b, hash_b = _oof(root, "R011")
    if meta_a != {**meta_a, "sha256": meta_a["sha256"]}: raise ValueError("Unreachable metadata guard")
    if any(meta_a[key] != meta_b[key] for key in ("identity_hash", "data_fingerprint", "class_order", "prediction_kind", "prediction_columns", "id_columns", "split_signature")):
        raise ValueError("R013/R011 metadata incompatible")
    if not a.transaction_id.equals(b.transaction_id) or not a.__row__.equals(b.__row__) or not a.__fold__.equals(b.__fold__) or a.transaction_id.duplicated().any() or b.transaction_id.duplicated().any():
        raise ValueError("R013/R011 OOF row, ID, or fold alignment failed")
    valid = a.__fold__.ge(0).to_numpy()
    if valid.sum() != 472173 or not a.loc[~valid, ["pred_0", "pred_1"]].isna().all().all() or not b.loc[~valid, ["pred_0", "pred_1"]].isna().all().all():
        raise ValueError("Unexpected warmup coverage")
    if not np.isfinite(a.loc[valid, ["pred_0", "pred_1"]].to_numpy()).all() or not np.isfinite(b.loc[valid, ["pred_0", "pred_1"]].to_numpy()).all():
        raise ValueError("Non-finite validation OOF prediction")
    config = Config(**r013["config"]); config.model = "ensemble"; config.model_params = {"members": list(MEMBERS), "weights": list(WEIGHTS), "method": "weighted"}; config.predict_test = False; config.aggregation = None
    hypothesis = "A fixed 25% LightGBM contribution improves R013 ranking robustness because the two models make partially different errors."
    change = "OOF-only arithmetic probability blend: 0.75*R013 CatBoost + 0.25*R011 LightGBM; exclude fold -1 warmup; no training, test inference, or weight search."
    record = reserve_experiment(root, "R016", config, hypothesis, change, parent="R013")
    try:
        provenance = root / record["provenance_path"]; record["source_hashes"] = snapshot_source(provenance / "source")
        save_json(provenance / "config.json", config.to_dict(), exclusive=True); save_json(provenance / "environment.json", environment(), exclusive=True)
        save_json(provenance / "splits.json", read_json(root / r013["splits_path"]), exclusive=True)
        raw = read_table(root / "train.csv", id_columns=["transaction_id"])
        if not a.transaction_id.equals(raw.transaction_id.astype("string")) or not np.array_equal(a.__row__.to_numpy(), np.arange(len(raw))): raise ValueError("OOF/train label alignment failed")
        blend = WEIGHTS[0] * a[["pred_0", "pred_1"]].to_numpy() + WEIGHTS[1] * b[["pred_0", "pred_1"]].to_numpy()
        if not np.allclose(blend[valid].sum(axis=1), 1, atol=1e-10, rtol=0): raise ValueError("Blend is not a probability matrix")
        assignment = a.__fold__.to_numpy(); fold_scores = [score(raw.loc[assignment == fold, "fraud"], blend[assignment == fold], config) for fold in range(2)]
        ts = pd.to_datetime(raw.timestamp, errors="raise"); diagnostics = {}
        for name, (begin, end) in WINDOWS.items():
            mask = (assignment == 1) & (ts >= begin) & (ts < end); diagnostics[name] = {"rows": int(mask.sum()), "positives": int(raw.loc[mask, "fraud"].sum()), "average_precision": score(raw.loc[mask, "fraud"], blend[mask], config)}
        pooled = score(raw.loc[valid, "fraud"], blend[valid], config)
        a_pos, b_pos = a.loc[valid, "pred_1"].to_numpy(), b.loc[valid, "pred_1"].to_numpy()
        correlation = {"pearson": float(pearsonr(a_pos, b_pos).statistic), "rank_spearman": float(pearsonr(rankdata(a_pos), rankdata(b_pos)).statistic)}
        oof_path = "outputs/oof/R016.csv"; fingerprint = meta_a["data_fingerprint"]; save_predictions(root / oof_path, a, blend, config, fingerprint, folds=assignment, split_signature=SPLIT)
        keys = {"F1": fold_scores[0], "F2": fold_scores[1], "mean": float(np.mean(fold_scores)), "pooled": pooled, **{name: item["average_precision"] for name, item in diagnostics.items()}}
        base = {"F1": r013["fold_scores"][0], "F2": r013["fold_scores"][1], "mean": r013["cv_mean"], "pooled": r013["pooled_covered_oof_score"], **{name: r013["diagnostics"][name]["average_precision"] for name in WINDOWS}}
        deltas = {key: float(keys[key] - base[key]) for key in keys}
        decision = "WIN" if deltas["F2"] >= .002 and deltas["F2_late_half"] >= 0 and deltas["July_1_15"] >= 0 else "LOSE" if deltas["F2_late_half"] < -.002 or deltas["July_1_15"] < -.002 else "FLAT-MIXED"
        record.update(ensemble=config.model_params, train_fingerprint=fingerprint, split_signature=SPLIT, splits_path=f"{record['provenance_path']}splits.json", oof_path=oof_path, fold_scores=fold_scores, cv_mean=keys["mean"], cv_std=float(np.std(fold_scores)), cv_std_definition="population std (ddof=0)", oof_coverage=float(valid.mean()), pooled_covered_oof_score=pooled, diagnostics=diagnostics, training_seconds=time.perf_counter()-start, status="completed", pre_fit_assertions={"r013_r011_oof_hashes_verified": {"R013": hash_a, "R011": hash_b}, "exact_row_id_fold_alignment": True, "warmup_excluded_from_scoring": True, "no_model_training": True, "no_test_inference": True, "fixed_weights": list(WEIGHTS)})
        save_json(provenance / "decision_report.json", {"experiment_id": "R016", "hypothesis": hypothesis, "metrics": keys, "deltas_vs_r013": deltas, "correlation": correlation, "decision": decision, "decision_rule": "WIN requires F2 >= +0.002 with nonnegative late and July; fixed weights were not optimized", "runtime_seconds": record["training_seconds"]}, exclusive=True)
        finish_record(root, record); print(f"R016 {decision}: F2 delta {deltas['F2']:.8f}; July delta {deltas['July_1_15']:.8f}")
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", training_seconds=time.perf_counter()-start); finish_record(root, record); raise


if __name__ == "__main__": main()
