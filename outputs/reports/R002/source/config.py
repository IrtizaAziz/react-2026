"""Edit DEFAULT_CONFIG after launch, or supply an equivalent JSON with --config."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    train_file: str | None = None
    test_file: str | None = None
    sample_file: str | None = None
    target: str | None = None
    id_columns: list[str] = field(default_factory=list)
    features: list[str] | None = None
    row_features: list[str] = field(default_factory=list)  # explicit deterministic features derived per row
    categorical_features: list[str] = field(default_factory=list)  # explicit native categorical features for supported models
    task: str | None = None  # regression | classification
    metric: str | None = None
    metric_direction: str | None = None  # higher | lower; check Evaluation
    metric_params: dict = field(default_factory=dict)
    custom_metric: str | None = None  # src.my_metric:score
    custom_metric_kind: str | None = None
    prediction_kind: str | None = None  # value | probability | label | decision
    class_order: list = field(default_factory=list)
    validation_type: str | None = None
    validation_rationale: str | None = None
    n_splits: int | None = None
    shuffle: bool = False
    seed: int = 42
    group_column: str | None = None
    time_column: str | None = None
    time_gap: int = 0
    time_valid_fraction: float = 0.2
    calendar_folds: list[dict] = field(default_factory=list)
    feature_profile: str | None = None
    execution_fold_order: list[int] = field(default_factory=list)
    diagnostic_windows: list[dict] = field(default_factory=list)
    debug_expected_fold_scores: dict = field(default_factory=dict)
    splits_file: str | None = None
    model: str | None = None
    model_params: dict = field(default_factory=dict)
    text_column: str | None = None
    predict_test: bool = False
    aggregation: str | None = None  # mean, explicitly requested for fold inference
    submission_columns: list[str] = field(default_factory=list)
    submission_kind: str | None = None  # value | label | probability
    submission_alignment: str | None = None  # id | position
    positive_class: str | int | float | None = None
    label_threshold: float | None = None
    smoke_test: bool = False

    def to_dict(self):
        return asdict(self)

    def path(self, root, value):
        if not value:
            raise ValueError("A required file path is not configured.")
        path = Path(value)
        return path if path.is_absolute() else Path(root) / path

    def validate(self):
        required = ["train_file", "target", "features", "task", "metric",
                    "metric_direction", "prediction_kind", "validation_type",
                    "validation_rationale", "n_splits", "model"]
        missing = [key for key in required if not getattr(self, key)]
        if missing:
            raise ValueError("Configure after launch before training: " + ", ".join(missing))
        if self.task not in {"regression", "classification"}:
            raise ValueError("task must be regression or classification")
        if self.metric_direction not in {"higher", "lower"}:
            raise ValueError("metric_direction must be higher or lower")
        if self.n_splits < (1 if self.validation_type == "time_holdout" else 2):
            raise ValueError("time_holdout requires n_splits=1; other validation requires at least 2 folds")
        if not 0 < self.time_valid_fraction < 1:
            raise ValueError("time_valid_fraction must be between 0 and 1")
        if len(set(self.features)) != len(self.features) or self.target in self.features:
            raise ValueError("Features must be unique and cannot include the target")
        supported_row_features = {"FamilySize", "IsAlone", "Title"}
        if not set(self.row_features) <= supported_row_features:
            raise ValueError(f"Unsupported row_features: {sorted(set(self.row_features) - supported_row_features)}")
        if not set(self.row_features) <= set(self.features):
            raise ValueError("row_features must be included in features")
        if not set(self.categorical_features) <= set(self.features):
            raise ValueError("categorical_features must be included in features")
        if len(set(self.id_columns)) != len(self.id_columns) or any(c in {"__row__", "__fold__"} or c.startswith("pred_") for c in self.id_columns):
            raise ValueError("ID columns must be unique and cannot use reserved artifact column names")
        if set(self.id_columns) & set(self.features):
            raise ValueError("ID columns cannot be features; review suspicious IDs explicitly first")
        if self.validation_type == "calendar_time":
            if self.shuffle or not self.time_column:
                raise ValueError("calendar_time requires time_column and shuffle=false")
            if len(self.calendar_folds) != self.n_splits:
                raise ValueError("calendar_time requires one explicit interval per fold")
            for fold in self.calendar_folds:
                if set(fold) != {"train_before", "valid_start", "valid_end"}:
                    raise ValueError("Each calendar fold requires train_before, valid_start, and valid_end")
        elif self.calendar_folds:
            raise ValueError("calendar_folds is only valid with validation_type='calendar_time'")
        if self.execution_fold_order and sorted(self.execution_fold_order) != list(range(self.n_splits)):
            raise ValueError("execution_fold_order must be a permutation of canonical fold indexes")
        for window in self.diagnostic_windows:
            if set(window) != {"name", "fold", "start", "end"} or not 0 <= window["fold"] < self.n_splits:
                raise ValueError("Each diagnostic window requires name, fold, start, and end")
        if any(not isinstance(key, str) or not key.startswith("F") or not 0 <= int(key[1:]) - 1 < self.n_splits
               for key in self.debug_expected_fold_scores):
            raise ValueError("debug_expected_fold_scores keys must name configured folds, such as F2")
        if self.feature_profile == "react2026_static":
            expected = {"amount_bdt", "log_amount_bdt", "account_age_days", "hour", "weekday", "is_weekend",
                        "merchant_category", "device_type", "location", "payment_method", "transaction_type",
                        "merchant_category_missing", "device_type_missing", "location_missing"}
            if set(self.features) != expected:
                raise ValueError("react2026_static requires exactly the approved static feature set")
            categories = {"merchant_category", "device_type", "location", "payment_method", "transaction_type"}
            if set(self.categorical_features) != categories:
                raise ValueError("react2026_static requires exactly the approved low-cardinality categoricals")
        elif self.feature_profile is not None:
            raise ValueError("Unknown feature_profile")
        if self.task == "classification":
            if len(self.class_order) < 2 or len(set(self.class_order)) != len(self.class_order):
                raise ValueError("Explicit, unique class_order with at least two classes is required")
            if self.prediction_kind not in {"probability", "label", "decision"}:
                raise ValueError("Classification requires probability, label, or decision predictions")
        elif self.prediction_kind != "value":
            raise ValueError("Regression requires value predictions")
        if self.predict_test and (not self.test_file or self.aggregation != "mean"):
            raise ValueError("Test inference requires test_file and explicit aggregation='mean'")
        if self.predict_test and self.prediction_kind == "label":
            raise ValueError("Fold inference requires numeric values, probabilities, or decision scores; use probabilities for classification labels")


DEFAULT_CONFIG = Config()


def load_config(path=None):
    if path is None:
        return Config(**DEFAULT_CONFIG.to_dict())
    with Path(path).open(encoding="utf-8-sig") as handle:
        return Config(**json.load(handle))


if __name__ == "__main__":
    print(json.dumps(DEFAULT_CONFIG.to_dict(), indent=2))
