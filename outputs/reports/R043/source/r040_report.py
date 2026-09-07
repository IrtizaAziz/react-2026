"""Canonical immutable OOF gate report for the fixed R029/R037 blend."""
from pathlib import Path
import json

from .config import Config, ROOT
from .ensemble import load_oof_scoring_frame, score_canonical_oof
from .predict import read_predictions
from .utils import read_json, save_json, sha256

MEMBERS = ("R029", "R037")
BASELINES = ("R029", "R037", "R017")
WEIGHTS = (0.75, 0.25)
WINDOWS = (("F2_early", 1, "2026-05-15", "2026-06-15"),
           ("F2_late_corrected", 1, "2026-06-15", "2026-07-16"),
           ("June_15_30", 1, "2026-06-15", "2026-07-01"),
           ("July", 1, "2026-07-01", "2026-07-16"),
           ("July_1_7", 1, "2026-07-01", "2026-07-08"),
           ("July_8_15", 1, "2026-07-08", "2026-07-16"))


def run(root=ROOT, experiment="R040"):
    root = Path(root)
    if experiment != "R040" or WEIGHTS != (.75, .25):
        raise ValueError("This report is locked to R040 = 0.75*R029 + 0.25*R037")
    records = {name: read_json(root / f"outputs/reports/{name}.json") for name in (*BASELINES, experiment)}
    if any(record.get("status") != "completed" for record in records.values()):
        raise ValueError("All member and R040 records must be completed")
    config = Config(**records["R029"]["config"])
    raw, fingerprint = load_oof_scoring_frame(root, config)
    metric_sets = {}
    artifact_hashes = {}
    for name, record in records.items():
        frame, values, metadata = read_predictions(root / record["oof_path"])
        if metadata["data_fingerprint"] != fingerprint or not frame["transaction_id"].equals(raw["transaction_id"].astype("string")):
            raise ValueError(f"{name} OOF/training identity mismatch")
        metric_sets[name] = score_canonical_oof(raw, values, frame["__fold__"].to_numpy(), config, WINDOWS)
        artifact_hashes[name] = sha256(root / record["oof_path"])
    metrics = metric_sets[experiment]
    deltas = {name: {key: metrics[key] - metric_sets[name][key] for key in metrics} for name in BASELINES}
    gates = {"F2": metrics["F2"] >= metric_sets["R017"]["F2"] + .003,
             "corrected_late": metrics["F2_late_corrected"] >= metric_sets["R017"]["F2_late_corrected"] + .002,
             "July": metrics["July"] >= metric_sets["R017"]["July"] + .003,
             "F1": metrics["F1"] >= metric_sets["R017"]["F1"] - .002,
             "July_1_7": metrics["July_1_7"] >= metric_sets["R017"]["July_1_7"] - .002,
             "July_8_15": metrics["July_8_15"] >= metric_sets["R017"]["July_8_15"] - .002}
    report = {"experiment_id": experiment, "hypothesis": "0.75 * R029 OOF + 0.25 * R037 OOF", "weights": list(WEIGHTS),
              "metrics": metric_sets, "deltas": deltas, "gates_vs_R017": gates,
              "classification": "CLEAR WIN" if all(gates.values()) else "FLAT-NO-SUBMISSION",
              "test_predictions": False, "submission": False, "oof_sha256": artifact_hashes,
              "scoring": "canonical validation-covered rows only; expected warmup rows remain null"}
    save_json(root / "outputs/reports/R040/r040_gate_report.json", report, exclusive=True)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
