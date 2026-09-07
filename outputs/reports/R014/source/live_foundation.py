"""Persist the reviewed REACT 2026 live input and calendar-split foundation; never train."""
if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
from pathlib import Path

from .config import ROOT, load_config
from .data import load_test, load_training, read_table
from .metrics import metric_definition
from .utils import save_json, sha256
from .validation import make_splits


def build_foundation(root=ROOT, *, config_path=None, data_dictionary="../data_dictionary.csv", output="reports/live_foundation.json", splits_output="outputs/reports/live_foundation_splits.json"):
    """Validate static live inputs and write immutable Stage 1 metadata only."""
    root = Path(root)
    config = load_config(config_path or root / "config_live_stage1.json")
    config.validate()
    metric_definition(config)
    train, train_fingerprint = load_training(config, root)
    test, test_fingerprint = load_test(config, root)
    sample_path = config.path(root, config.sample_file)
    sample = read_table(sample_path, id_columns=config.id_columns)
    dictionary_path = config.path(root, data_dictionary)
    if list(sample.columns) != ["transaction_id", "fraud"] or not sample["transaction_id"].equals(test["transaction_id"]):
        raise ValueError("Live sample schema/order must exactly match test transaction IDs and transaction_id,fraud")
    pairs, assignment, splits = make_splits(train, config, train_fingerprint, save_to=root / splits_output)
    expected_counts = [(259769, 232108), (491877, 240065)]
    actual_counts = [(len(train_idx), len(valid_idx)) for train_idx, valid_idx in pairs]
    if actual_counts != expected_counts:
        raise ValueError(f"Live calendar split counts differ: expected {expected_counts}, found {actual_counts}")
    payload = {
        "stage": "live_foundation",
        "training_performed": False,
        "experiment_id_reserved": None,
        "config": config.to_dict(),
        "input_hashes": {
            "train.csv": sha256(config.path(root, config.train_file)),
            "test.csv": sha256(config.path(root, config.test_file)),
            "sample_submission.csv": sha256(sample_path),
            "data_dictionary.csv": sha256(dictionary_path),
        },
        "input_fingerprints": {"train": train_fingerprint, "test": test_fingerprint,
                               "sample": {"rows": len(sample), "columns": list(sample.columns)}},
        "split_signature": splits["signature"],
        "splits_path": splits_output,
        "fold_counts": [{"fold": i + 1, "train_rows": len(train_idx), "valid_rows": len(valid_idx)}
                        for i, (train_idx, valid_idx) in enumerate(pairs)],
        "warmup_rows": int((assignment < 0).sum()),
    }
    save_json(root / output, payload, exclusive=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dictionary", default="../data_dictionary.csv")
    args = parser.parse_args()
    try:
        result = build_foundation(args.root, config_path=args.config, data_dictionary=args.data_dictionary)
        print(result["split_signature"])
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Live foundation stopped: {exc}\n")


if __name__ == "__main__":
    main()
