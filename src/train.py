"""Explicit CV training. Test inference happens only after all fitting is complete."""
if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
from sklearn.base import clone
from .config import ROOT, load_config
from .data import load_training
from .features import build_pipeline
from .metrics import metric_definition, score
from .predict import model_predictions, predict_experiment, save_predictions
from .utils import environment, finish_record, reserve_experiment, save_json, seed_everything, sha256, snapshot_source
from .validation import make_splits


def train_experiment(config, experiment, hypothesis, change, *, root=ROOT, parent=None):
    root = Path(root)
    config.validate()
    metric_definition(config)
    template = build_pipeline(config)  # Optional dependencies checked before reserving the ID.
    frame, fingerprint = load_training(config, root)
    reuse = config.path(root, config.splits_file) if config.splits_file else None
    pairs, assignment, splits = make_splits(frame, config, fingerprint, reuse=reuse)
    if config.task == "classification":
        if set(frame[config.target]) != set(config.class_order):
            raise ValueError("class_order does not match the training labels")
        encoded_y = frame[config.target].map({label: i for i, label in enumerate(config.class_order)})
    else:
        encoded_y = frame[config.target]
    record = reserve_experiment(root, experiment, config, hypothesis, change, parent)
    start = time.perf_counter()
    try:
        seed_everything(config.seed)
        provenance = root / record["provenance_path"]
        record["source_hashes"] = snapshot_source(provenance / "source")
        save_json(provenance / "config.json", config.to_dict(), exclusive=True)
        save_json(provenance / "environment.json", environment(used_packages=[config.model]), exclusive=True)
        save_json(provenance / "splits.json", splits, exclusive=True)
        record.update(train_fingerprint=fingerprint, split_signature=splits["signature"],
                      splits_path=f"{record['provenance_path']}splits.json", model_paths=[], model_hashes={})
        scores, oof, fitted_parameters = [], None, []
        model_dir = root / "outputs/models" / experiment
        model_dir.mkdir(parents=True, exist_ok=False)
        log = []
        for fold, (training, valid) in enumerate(pairs):
            pipeline = clone(template)
            pipeline.fit(frame.iloc[training][config.features], encoded_y.iloc[training])
            predictions = np.asarray(model_predictions(pipeline, frame.iloc[valid][config.features], config))
            if oof is None:
                shape = (len(frame), *predictions.shape[1:])
                oof = np.full(shape, None, dtype=object) if config.prediction_kind == "label" else np.full(shape, np.nan)
            oof[valid] = predictions
            fold_score = score(frame.iloc[valid][config.target], predictions, config)
            scores.append(fold_score)
            message = f"{experiment} fold {fold + 1}/{len(pairs)}: {fold_score:.8g}"
            print(message)
            log.append(message)
            relative = f"outputs/models/{experiment}/fold_{fold}.pkl"
            with (root / relative).open("xb") as handle:
                pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
            record["model_paths"].append(relative)
            record["model_hashes"][relative] = sha256(root / relative)
            fitted_parameters.append(pipeline.named_steps["model"].get_params())
        oof_path = f"outputs/oof/{experiment}.csv"
        save_predictions(root / oof_path, frame, oof, config, fingerprint, folds=assignment, split_signature=splits["signature"])
        record.update(status="completed", fold_scores=scores, cv_mean=float(np.mean(scores)),
                      cv_std=float(np.std(scores, ddof=0)), cv_std_definition="population std (ddof=0)",
                      oof_coverage=float(np.mean(assignment >= 0)), oof_path=oof_path,
                      training_seconds=time.perf_counter() - start,
                      fitted_model_parameters=fitted_parameters)
        save_json(provenance / "model_parameters.json", fitted_parameters, exclusive=True)
        (provenance / "training.log").write_text("\n".join(log) + "\n", encoding="utf-8")
        finish_record(root, record)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                      training_seconds=time.perf_counter() - start)
        finish_record(root, record)
        raise
    if config.predict_test:
        try:
            record = predict_experiment(root, experiment, aggregation=config.aggregation)
        except Exception as exc:
            record["inference_error"] = f"{type(exc).__name__}: {exc}"
            finish_record(root, record)
            raise
    print(f"CV {record['cv_mean']:.8g} +/- {record['cv_std']:.8g}; OOF coverage {record['oof_coverage']:.1%}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--change", required=True)
    parser.add_argument("--parent")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--predict-test", action="store_true", default=None)
    group.add_argument("--no-test-inference", dest="predict_test", action="store_false")
    parser.add_argument("--aggregation", choices=["mean"])
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        for key in ("model", "seed", "predict_test", "aggregation"):
            if getattr(args, key) is not None:
                setattr(config, key, getattr(args, key))
        train_experiment(config, args.experiment, args.hypothesis, args.change, root=args.root, parent=args.parent)
    except (ValueError, OSError, ImportError, TypeError) as exc:
        parser.exit(2, f"Training stopped: {exc}\n")


if __name__ == "__main__":
    main()
