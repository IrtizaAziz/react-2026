"""Explicit arithmetic/weighted blends evaluated on compatible OOF predictions."""
if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
from pathlib import Path
import time
import numpy as np
from scipy.stats import rankdata
from .config import Config, ROOT
from .data import load_training
from .metrics import score
from .predict import read_predictions, save_predictions
from .utils import checked_record, environment, finish_record, read_json, reserve_experiment, save_json, snapshot_source

COMPATIBLE_FIELDS = ("target", "task", "metric", "metric_direction", "metric_params", "custom_metric", "custom_metric_kind",
                     "prediction_kind", "class_order", "positive_class", "label_threshold", "id_columns")


def rank_average(items, config):
    """Average within-column ranks only for explicitly ranking-compatible metrics."""
    if config.metric not in {"roc_auc", "average_precision"}:
        raise ValueError("Rank averaging is reserved for ranking metrics; use mean/weighted for this metric")
    arrays = [np.asarray(item, dtype=float) for item in items]
    arrays = [a[:, None] if a.ndim == 1 else a for a in arrays]
    ranked = [np.column_stack([rankdata(a[:, i], method="average") for i in range(a.shape[1])]) for a in arrays]
    result = np.mean(ranked, axis=0)
    if config.prediction_kind == "probability":
        result = result / result.sum(axis=1, keepdims=True)
    return result[:, 0] if config.prediction_kind == "decision" else result


def compatible_artifacts(items):
    base_frame, base_values, base = items[0]
    for frame, values, metadata in items[1:]:
        for key in ("identity_hash", "data_fingerprint", "class_order", "prediction_kind", "prediction_columns", "split_signature", "id_columns"):
            if metadata[key] != base[key]:
                raise ValueError(f"Incompatible ensemble artifacts: {key}")
        if values.shape != base_values.shape:
            raise ValueError("Incompatible prediction shapes")
        if "__fold__" in base_frame and not np.array_equal(base_frame["__fold__"], frame["__fold__"]):
            raise ValueError("Incompatible OOF coverage/fold assignment")
    return base_frame, base


def blend_experiments(root, experiment, members, weights, hypothesis, change, *, method="weighted", predict_test=False, train_file=None):
    root = Path(root)
    weights = np.asarray(weights, dtype=float)
    if len(members) < 2 or len(set(members)) != len(members) or len(weights) != len(members):
        raise ValueError("Provide at least two distinct member IDs and one weight per member")
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1, atol=1e-8, rtol=0):
        raise ValueError("Weights must be finite, nonnegative, and sum to one; they are never automatically normalized")
    if method not in {"mean", "weighted", "rank"} or (method == "mean" and not np.allclose(weights, 1/len(members), rtol=0, atol=1e-8)):
        raise ValueError("Method must be mean, weighted, or rank; arithmetic mean requires equal weights")
    if method == "rank" and not np.allclose(weights, 1/len(members), rtol=0, atol=1e-8):
        raise ValueError("Rank averaging uses equal member weights")
    records = [checked_record(root, member) for member in members]
    config = Config(**records[0]["config"])
    if config.prediction_kind == "label":
        raise ValueError("Do not average class labels; use compatible probabilities")
    for record in records[1:]:
        if any(record["config"][key] != config.to_dict()[key] for key in COMPATIBLE_FIELDS):
            raise ValueError("Ensemble task/metric/class/decision configuration mismatch")
        if record["split_signature"] != records[0]["split_signature"]:
            raise ValueError("Ensemble members must use identical saved splits")
    oofs = [read_predictions(root / record["oof_path"]) for record in records]
    oof_frame, oof_meta = compatible_artifacts(oofs)
    test_items = []
    if predict_test:
        if any(not r.get("prediction_path") for r in records):
            raise ValueError("All members need explicit test inference before blending predictions")
        test_items = [read_predictions(root / r["prediction_path"]) for r in records]
        compatible_artifacts(test_items)
    if train_file:
        config.train_file = train_file
    frame, fingerprint = load_training(config, root)
    if fingerprint != oof_meta["data_fingerprint"]:
        raise ValueError("Training labels no longer match saved OOF data")
    config.model = "ensemble"
    config.model_params = {"members": members, "weights": weights.tolist(), "method": method}
    config.predict_test = predict_test
    config.aggregation = "mean" if predict_test else None
    record = reserve_experiment(root, experiment, config, hypothesis, change)
    start = time.perf_counter()
    try:
        provenance = root / record["provenance_path"]
        record["source_hashes"] = snapshot_source(provenance / "source")
        save_json(provenance / "config.json", config.to_dict(), exclusive=True)
        save_json(provenance / "environment.json", environment(), exclusive=True)
        split_payload = read_json(root / records[0]["splits_path"])
        save_json(provenance / "splits.json", split_payload, exclusive=True)
        blended = rank_average([item[1] for item in oofs], config) if method == "rank" else sum(weight * item[1] for weight, item in zip(weights, oofs))
        assignment = oof_frame["__fold__"].to_numpy()
        fold_scores = [score(frame.loc[assignment == fold, config.target], blended[assignment == fold], config) for fold in range(config.n_splits)]
        oof_path = f"outputs/oof/{experiment}.csv"
        save_predictions(root / oof_path, frame, blended, config, fingerprint, folds=assignment, split_signature=oof_meta["split_signature"])
        record.update(ensemble=config.model_params, train_fingerprint=fingerprint, split_signature=oof_meta["split_signature"],
                      splits_path=f"{record['provenance_path']}splits.json", oof_path=oof_path,
                      fold_scores=fold_scores, cv_mean=float(np.mean(fold_scores)), cv_std=float(np.std(fold_scores)),
                      cv_std_definition="population std (ddof=0)", oof_coverage=float(np.mean(assignment >= 0)))
        if test_items:
            test_frame, test_meta = compatible_artifacts(test_items)
            predictions = rank_average([item[1] for item in test_items], config) if method == "rank" else sum(weight * item[1] for weight, item in zip(weights, test_items))
            path = f"outputs/predictions/{experiment}.csv"
            save_predictions(root / path, test_frame, predictions, config, test_meta["data_fingerprint"])
            record.update(prediction_path=path, test_fingerprint=test_meta["data_fingerprint"])
        record.update(status="completed", training_seconds=time.perf_counter()-start)
        finish_record(root, record)
        print(f"{experiment}: CV {record['cv_mean']:.8g} +/- {record['cv_std']:.8g}")
        return record
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finish_record(root, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--method", choices=["mean", "weighted", "rank"], default="weighted")
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--change", required=True)
    parser.add_argument("--predict-test", action="store_true")
    args = parser.parse_args()
    try:
        blend_experiments(args.root, args.experiment, args.members, args.weights, args.hypothesis, args.change,
                          method=args.method, predict_test=args.predict_test)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Ensemble stopped: {exc}\n")


if __name__ == "__main__":
    main()
