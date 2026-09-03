"""Modest baseline pipelines. All learned transforms are fitted inside folds."""
import importlib
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor


def clean_categories(frame):
    return frame.astype(object).where(frame.notna(), np.nan).map(lambda x: str(x) if x is not np.nan and x is not None and not (isinstance(x, float) and np.isnan(x)) else np.nan)


def text_values(frame):
    return frame.iloc[:, 0].fillna("").astype(str).to_numpy()


def estimator(config):
    classifier = config.task == "classification"
    params = dict(config.model_params)
    if config.model in {"linear", "logistic", "tfidf_logistic"}:
        if config.model != "linear" and not classifier:
            raise ValueError("Logistic models require classification")
        if classifier:
            return LogisticRegression(**{"max_iter": 1000, "random_state": config.seed, **params})
        return LinearRegression(**params)
    if config.model == "tfidf_linear":
        if not classifier:
            raise ValueError("tfidf_linear is a classification template")
        if config.prediction_kind == "probability":
            raise ValueError("LinearSVC has no predict_proba; choose another model or explicitly design calibration")
        return LinearSVC(**{"random_state": config.seed, **params})
    if config.model in {"random_forest", "extra_trees"}:
        cls = (RandomForestClassifier if classifier else RandomForestRegressor) if config.model == "random_forest" else (ExtraTreesClassifier if classifier else ExtraTreesRegressor)
        return cls(**{"n_estimators": 100, "random_state": config.seed, "n_jobs": 1, **params})
    choices = {"catboost": ("catboost", "CatBoost", {"random_seed": config.seed, "verbose": False, "allow_writing_files": False}),
               "lightgbm": ("lightgbm", "LGBM", {"random_state": config.seed, "verbosity": -1, "n_jobs": 1}),
               "xgboost": ("xgboost", "XGB", {"random_state": config.seed, "n_jobs": 1})}
    if config.model not in choices:
        raise ValueError(f"Unknown model {config.model!r}")
    package, prefix, defaults = choices[config.model]
    try:
        module = importlib.import_module(package)
    except ImportError as exc:
        raise ImportError(f"Optional model {config.model} is not installed. Install explicitly: pip install {package}") from exc
    cls = getattr(module, prefix + ("Classifier" if classifier else "Regressor"))
    return cls(**{**defaults, **params})


def build_pipeline(config):
    model = estimator(config)
    if config.model.startswith("tfidf_"):
        if config.features != [config.text_column]:
            raise ValueError("TF-IDF templates require features=[text_column]")
        return Pipeline([("text", FunctionTransformer(text_values)), ("tfidf", TfidfVectorizer()), ("model", model)])
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler(with_mean=False))])
    categorical = Pipeline([("clean", FunctionTransformer(clean_categories)),
                            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
                            ("encode", OneHotEncoder(handle_unknown="ignore", dtype=np.float64))])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number))])
    return Pipeline([("preprocess", preprocessing), ("model", model)])
