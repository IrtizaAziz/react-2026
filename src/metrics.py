"""Helpers, not a declaration of the official competition metric."""
from dataclasses import dataclass
import importlib
import numpy as np
from sklearn import metrics as skm


@dataclass(frozen=True)
class Metric:
    function: object
    direction: str
    kind: str


REGISTRY = {
    "mae": Metric(skm.mean_absolute_error, "lower", "value"),
    "mse": Metric(skm.mean_squared_error, "lower", "value"),
    "rmse": Metric(lambda y, p, **kw: np.sqrt(skm.mean_squared_error(y, p, **kw)), "lower", "value"),
    "r2": Metric(skm.r2_score, "higher", "value"),
    "accuracy": Metric(skm.accuracy_score, "higher", "label"),
    "f1": Metric(skm.f1_score, "higher", "label"),
    "precision": Metric(skm.precision_score, "higher", "label"),
    "recall": Metric(skm.recall_score, "higher", "label"),
    "log_loss": Metric(skm.log_loss, "lower", "probability"),
    "roc_auc": Metric(skm.roc_auc_score, "higher", "score"),
    "average_precision": Metric(skm.average_precision_score, "higher", "score"),
}


def metric_definition(config):
    if config.metric == "custom":
        if not config.custom_metric or config.custom_metric_kind not in {"value", "label", "probability", "decision"}:
            raise ValueError("Custom metric requires module:function and custom_metric_kind")
        module, name = config.custom_metric.split(":", 1)
        metric = Metric(getattr(importlib.import_module(module), name), config.metric_direction, config.custom_metric_kind)
    else:
        if config.metric not in REGISTRY:
            raise ValueError(f"Unknown metric {config.metric!r}; add the exact official implementation")
        metric = REGISTRY[config.metric]
    if metric.direction != config.metric_direction:
        raise ValueError("Metric direction disagrees with the helper; implement a custom metric if Evaluation transforms it")
    if config.metric in {"f1", "precision", "recall"}:
        if "average" not in config.metric_params:
            raise ValueError("Set metric_params.average explicitly")
        if config.metric_params["average"] == "binary" and "pos_label" not in config.metric_params:
            raise ValueError("Binary averaging requires explicit metric_params.pos_label")
    compatible = {"value": {"value"}, "label": {"label", "probability"},
                  "probability": {"probability"}, "score": {"probability", "decision"},
                  "decision": {"decision"}}
    if config.prediction_kind not in compatible[metric.kind]:
        raise ValueError(f"Metric {config.metric} cannot consume {config.prediction_kind} predictions")
    return metric


def labels_from_probabilities(predictions, config):
    predictions = np.asarray(predictions)
    if len(config.class_order) == 2:
        if config.positive_class not in config.class_order or config.label_threshold is None or not 0 <= config.label_threshold <= 1:
            raise ValueError("Binary label conversion requires positive_class and label_threshold in [0, 1]")
        positive = config.class_order.index(config.positive_class)
        indices = np.where(predictions[:, positive] >= config.label_threshold, positive, 1 - positive)
    else:
        indices = predictions.argmax(axis=1)
    return np.asarray(config.class_order)[indices]


def score(y, predictions, config):
    metric = metric_definition(config)
    y, predictions = np.asarray(y), np.asarray(predictions)
    if config.prediction_kind != "label" and not np.isfinite(predictions.astype(float)).all():
        raise ValueError("Non-finite predictions cannot be scored")
    if config.prediction_kind == "probability":
        if predictions.shape != (len(y), len(config.class_order)) or (predictions < 0).any() or (predictions > 1).any() or not np.allclose(predictions.sum(axis=1), 1, atol=1e-6, rtol=0):
            raise ValueError("Metric requires valid probabilities in the configured class order")
    params = dict(config.metric_params)
    if metric.kind == "label" and config.prediction_kind == "probability":
        predictions = labels_from_probabilities(predictions, config)
    if config.metric == "log_loss":
        mapping = {label: i for i, label in enumerate(config.class_order)}
        y = np.array([mapping[label] for label in y])
        params["labels"] = np.arange(len(config.class_order))
    elif config.metric in {"roc_auc", "average_precision"}:
        if len(config.class_order) == 2:
            if config.positive_class not in config.class_order:
                raise ValueError("Binary ranking metrics require an explicit positive_class")
            positive = config.class_order.index(config.positive_class)
            y = (y == config.positive_class).astype(int)
            if config.prediction_kind == "probability":
                predictions = predictions[:, positive]
            elif positive == 0:
                predictions = -predictions
        else:
            if config.metric != "roc_auc" or config.prediction_kind != "probability":
                raise ValueError("This helper supports multiclass ROC AUC with probabilities; otherwise add a custom metric")
            if not {"average", "multi_class"} <= params.keys():
                raise ValueError("Multiclass ROC AUC needs explicit average and multi_class parameters")
            mapping = {label: i for i, label in enumerate(config.class_order)}
            y = np.array([mapping[label] for label in y])
            params["labels"] = np.arange(len(config.class_order))
    value = float(metric.function(y, predictions, **params))
    if not np.isfinite(value):
        raise ValueError("Metric is undefined for this fold; investigate validation instead of ignoring the fold")
    return value
