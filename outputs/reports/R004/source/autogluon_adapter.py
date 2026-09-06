"""Optional AutoGluon challenger boundary.

AutoGluon scores are explicitly non-comparable unless its folds are supplied
through the same locked validation design. It never owns submission creation.
"""
from pathlib import Path


def require_autogluon():
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError("AutoGluon is optional; install it in a separate environment with requirements-optional.txt") from exc
    return TabularPredictor


def fit_challenger(train, target, *, metric, path, time_limit=300, presets="medium_quality", num_cpus=1, num_gpus=0,
                   included_model_types=None, excluded_model_types=None):
    """Fit/persist an AutoGluon challenger and return its leaderboard.

    This function deliberately accepts a caller-provided training frame only;
    callers must establish and document the primary CV independently.
    """
    Predictor = require_autogluon()
    predictor = Predictor(label=target, eval_metric=metric, path=str(Path(path)), verbosity=0)
    predictor.fit(train_data=train, time_limit=time_limit, presets=presets, num_cpus=num_cpus, num_gpus=num_gpus,
                  included_model_types=included_model_types, excluded_model_types=excluded_model_types)
    leaderboard = predictor.leaderboard(train, silent=True)
    leaderboard["validation_comparability"] = "NON-COMPARABLE unless matched locked folds were used"
    return predictor, leaderboard
