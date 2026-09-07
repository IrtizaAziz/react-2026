"""Cheap matched-fold screening; full CV remains authoritative."""
if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
import copy
from pathlib import Path
import numpy as np
from sklearn.base import clone
from .config import Config, ROOT
from .data import load_training
from .features import build_pipeline
from .metrics import score
from .utils import checked_record, read_json


def screen_model(root, config, *, splits_file, fold=0):
    """Fit one fixed fold without reserving an experiment or writing artifacts."""
    root = Path(root)
    frame, _ = load_training(config, root)
    payload = read_json(root / splits_file)
    if fold < 0 or fold >= len(payload["splits"]):
        raise ValueError("screening fold is outside the saved split set")
    pair = payload["splits"][fold]
    template = build_pipeline(config)
    model = copy.deepcopy(template) if config.model == "catboost" else clone(template)
    model.fit(frame.iloc[pair["train"]][config.features], frame.iloc[pair["train"]][config.target])
    predictions = model.predict_proba(frame.iloc[pair["valid"]][config.features]) if config.prediction_kind == "probability" else model.predict(frame.iloc[pair["valid"]][config.features])
    return float(score(frame.iloc[pair["valid"]][config.target], np.asarray(predictions), config))


def screen_against_record(root, experiment, *, model=None, fold=0):
    root = Path(root)
    record = checked_record(root, experiment)
    config = Config(**record["config"])
    if model:
        config.model = model
    score_value = screen_model(root, config, splits_file=record["splits_path"], fold=fold)
    return {"experiment": experiment, "fold": fold, "screening_score": score_value,
            "warning": "Screening is directional only; run full CV before promotion."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    try:
        print(screen_against_record(args.root, args.experiment, model=args.model, fold=args.fold))
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Screening stopped: {exc}\n")


if __name__ == "__main__":
    main()
