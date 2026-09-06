"""Explicit CV training. Test inference happens only after all fitting is complete."""
if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
import copy
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.base import clone
from .config import ROOT, load_config
from .data import REACT2026_CUSTOMER_HISTORY_FEATURES, REACT2026_CUSTOMER_RELATIONSHIP_FEATURES, REACT2026_DEVICE_GLOBAL_FEATURES, REACT2026_STATIC_FEATURES, load_training
from .features import build_pipeline
from .metrics import metric_definition, score
from .predict import model_predictions, predict_experiment, save_predictions
from .replay import replay_fold_score
from .utils import environment, finish_record, reserve_experiment, save_json, seed_everything, sha256, snapshot_source
from .validation import make_splits


def assert_static_contract(config, frame, pairs):
    """Fail before fitting if the reviewed R001 static-only contract is violated."""
    if config.feature_profile not in {"react2026_static", "react2026_static_customer_history", "react2026_static_customer_history_relationships", "react2026_static_customer_history_relationships_device_global"}:
        return {}
    forbidden = {"transaction_id", "customer_id", "device_id", "merchant_id", "timestamp", config.target}
    expected = set(REACT2026_STATIC_FEATURES)
    if config.feature_profile in {"react2026_static_customer_history", "react2026_static_customer_history_relationships", "react2026_static_customer_history_relationships_device_global"}:
        expected.update(REACT2026_CUSTOMER_HISTORY_FEATURES)
    if config.feature_profile in {"react2026_static_customer_history_relationships", "react2026_static_customer_history_relationships_device_global"}:
        expected.update(REACT2026_CUSTOMER_RELATIONSHIP_FEATURES)
    if config.feature_profile == "react2026_static_customer_history_relationships_device_global":
        expected.update(REACT2026_DEVICE_GLOBAL_FEATURES)
    if set(config.features) != expected or forbidden & set(config.features):
        raise ValueError("Static feature contract includes an unexpected or prohibited feature")
    if {"class_weights", "scale_pos_weight", "auto_class_weights"} & set(config.model_params):
        raise ValueError("R001 static baseline forbids class weighting")
    cardinalities = []
    for training, _ in pairs:
        values = {column: int(frame.iloc[training][column].nunique(dropna=False)) for column in config.categorical_features}
        if any(value > 64 for value in values.values()):
            raise ValueError("A static categorical exceeds the approved 64-category limit in a training fold")
        cardinalities.append(values)
    return {"feature_profile": config.feature_profile, "features": list(config.features),
            "prohibited_features_absent": True,
            "historical_features_absent": config.feature_profile == "react2026_static",
            "approved_customer_history_only": config.feature_profile == "react2026_static_customer_history",
            "approved_customer_history_and_relationships_only": config.feature_profile == "react2026_static_customer_history_relationships",
            "approved_customer_history_relationships_and_device_global_only": config.feature_profile == "react2026_static_customer_history_relationships_device_global",
            "class_weighting_absent": True, "categorical_cardinality_by_fold": cardinalities,
            "preprocessing_fit_scope": "training rows only"}


def _feature_importance(pipeline, feature_names):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "get_feature_importance"):
        return []
    # The CatBoost preprocessor preserves the explicit numeric-then-categorical
    # config order. Avoid sklearn feature-name introspection because the small
    # category-cleaning FunctionTransformer intentionally has no name adapter.
    names = list(feature_names)
    return [{"feature": name, "importance": float(value)} for name, value in
            sorted(zip(names, model.get_feature_importance()), key=lambda item: item[1], reverse=True)]


def _catboost_model_feature_order(config):
    """Return the reviewed ColumnTransformer output order for CatBoost only."""
    categorical = list(config.categorical_features)
    return [column for column in config.features if column not in categorical] + categorical


def fold_recency_weights(timestamps, validation_start):
    """Normalized 60-day exponential supervised weights; never used by features or scoring."""
    times = pd.to_datetime(timestamps, errors="raise", utc=True)
    anchor = pd.Timestamp(validation_start, tz="UTC")
    age_days = (anchor - times).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    if (age_days < 0).any(): raise ValueError("Training timestamps cannot be after validation start")
    weights = np.exp2(-age_days / 60.0); weights /= weights.mean()
    if not np.isfinite(weights).all() or (weights <= 0).any() or not np.isclose(weights.mean(), 1.0): raise ValueError("Invalid normalized recency weights")
    return weights


def train_experiment(config, experiment, hypothesis, change, *, root=ROOT, parent=None):
    root = Path(root)
    config.validate()
    metric_definition(config)
    template = build_pipeline(config)  # Optional dependencies checked before reserving the ID.
    frame, fingerprint = load_training(config, root)
    reuse = config.path(root, config.splits_file) if config.splits_file else None
    pairs, assignment, splits = make_splits(frame, config, fingerprint, reuse=reuse)
    pre_fit_assertions = assert_static_contract(config, frame, pairs)
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
        record.update(train_fingerprint=fingerprint, split_signature=splits["signature"], pre_fit_assertions=pre_fit_assertions,
                      customer_history_generation_seconds=frame.attrs.get("customer_history_generation_seconds"),
                      customer_relationship_generation_seconds=frame.attrs.get("customer_relationship_generation_seconds"),
                      device_global_generation_seconds=frame.attrs.get("device_global_generation_seconds"),
                      splits_path=f"{record['provenance_path']}splits.json", model_paths=[], model_hashes={})
        scores, oof, fitted_parameters = [None] * len(pairs), None, [None] * len(pairs)
        fold_runtime_seconds, fold_prediction_paths, feature_importances, fold_replays = [None] * len(pairs), {}, {}, {}
        model_dir = root / "outputs/models" / experiment
        model_dir.mkdir(parents=True, exist_ok=False)
        log = []
        execution_order = config.execution_fold_order or list(range(len(pairs)))
        for fold in execution_order:
            training, valid = pairs[fold]
            # CatBoost normalizes cat_features internally, which violates sklearn.clone's
            # constructor-parameter identity check. A deep copy remains unfitted and keeps
            # each fold isolated; all other estimators retain standard sklearn cloning.
            pipeline = copy.deepcopy(template) if config.model == "catboost" else clone(template)
            fold_start = time.perf_counter()
            weights = None
            if config.supervised_weighting == "exponential_recent_60d":
                valid_start = config.calendar_folds[fold]["valid_start"]
                weights = fold_recency_weights(frame.iloc[training][config.time_column], valid_start)
                if np.any(np.diff(weights[np.argsort(pd.to_datetime(frame.iloc[training][config.time_column]).to_numpy())]) < 0):
                    raise ValueError("Newer rows must receive greater or equal recency weight")
            pipeline.fit(frame.iloc[training][config.features], encoded_y.iloc[training], **({"model__sample_weight": weights} if weights is not None else {}))
            predictions = np.asarray(model_predictions(pipeline, frame.iloc[valid][config.features], config))
            if config.prediction_kind == "probability" and (predictions.shape != (len(valid), len(config.class_order)) or not np.isfinite(predictions).all()):
                raise ValueError("Positive-class probability mapping or prediction shape is invalid")
            if oof is None:
                shape = (len(frame), *predictions.shape[1:])
                oof = np.full(shape, None, dtype=object) if config.prediction_kind == "label" else np.full(shape, np.nan)
            oof[valid] = predictions
            fold_score = score(frame.iloc[valid][config.target], predictions, config)
            scores[fold] = fold_score
            fold_runtime_seconds[fold] = time.perf_counter() - fold_start
            message = f"{experiment} canonical fold {fold + 1}/{len(pairs)}: {fold_score:.8g}"
            print(message)
            log.append(message)
            relative = f"outputs/models/{experiment}/fold_{fold}.pkl"
            with (root / relative).open("xb") as handle:
                pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
            record["model_paths"].append(relative)
            record["model_hashes"][relative] = sha256(root / relative)
            fitted_parameters[fold] = pipeline.named_steps["model"].get_params()
            feature_importances[f"F{fold + 1}"] = _feature_importance(
                pipeline,
                _catboost_model_feature_order(config) if config.model == "catboost" else config.features,
            )
            fold_path = f"outputs/oof/{experiment}/F{fold + 1}.csv"
            save_predictions(root / fold_path, frame.iloc[valid], predictions, config, fingerprint,
                             folds=np.full(len(valid), fold, dtype=int), split_signature=splits["signature"], row_numbers=valid)
            fold_prediction_paths[f"F{fold + 1}"] = fold_path
            replay = replay_fold_score(root / fold_path, root, config, expected_fold=fold, split_signature=splits["signature"])
            if not np.isclose(replay["average_precision"], fold_score, rtol=1e-9, atol=1e-12):
                raise ValueError("Saved fold prediction replay differs from the in-memory fold score")
            reference = config.debug_expected_fold_scores.get(f"F{fold + 1}")
            if reference is not None and not np.isclose(fold_score, float(reference), rtol=0, atol=1e-6):
                raise ValueError(f"Fold F{fold + 1} differs materially from the approved debugging reference")
            fold_replays[f"F{fold + 1}"] = replay
            if weights is not None:
                record.setdefault("supervised_weight_diagnostics", {})[f"F{fold + 1}"] = {"validation_start": config.calendar_folds[fold]["valid_start"], "half_life_days": 60, "min": float(weights.min()), "median": float(np.median(weights)), "mean": float(weights.mean()), "max": float(weights.max()), "finite": bool(np.isfinite(weights).all()), "strictly_positive": bool((weights > 0).all())}
        oof_path = f"outputs/oof/{experiment}.csv"
        save_predictions(root / oof_path, frame, oof, config, fingerprint, folds=assignment, split_signature=splits["signature"])
        covered = assignment >= 0
        pooled_covered_oof_score = score(frame.loc[covered, config.target], oof[covered], config)
        diagnostics = {}
        timestamps = np.asarray(frame[config.time_column].astype("string")) if config.time_column else None
        for window in config.diagnostic_windows:
            mask = ((assignment == window["fold"]) & (timestamps >= window["start"]) & (timestamps < window["end"]))
            if not mask.any():
                raise ValueError(f"Diagnostic window {window['name']} contains no rows")
            diagnostics[window["name"]] = {"fold": window["fold"], "rows": int(mask.sum()),
                                           "positives": int(frame.loc[mask, config.target].sum()),
                                           "average_precision": float(score(frame.loc[mask, config.target], oof[mask], config))}
        working_score = float(0.3 * scores[0] + 0.7 * scores[1]) if config.validation_type == "calendar_time" and len(scores) == 2 else None
        record.update(status="completed", fold_scores=scores, cv_mean=float(np.mean(scores)),
                      cv_std=float(np.std(scores, ddof=0)), cv_std_definition="population std (ddof=0)",
                      oof_coverage=float(np.mean(assignment >= 0)), oof_path=oof_path,
                      pooled_covered_oof_score=float(pooled_covered_oof_score),
                      fold_prediction_paths=fold_prediction_paths, fold_runtime_seconds=fold_runtime_seconds,
                      feature_importances=feature_importances, fold_replays=fold_replays, diagnostics=diagnostics,
                      working_score={"name": "S", "formula": "0.3*F1 + 0.7*F2", "value": working_score,
                                     "purpose": "internal model-selection heuristic"} if working_score is not None else None,
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
