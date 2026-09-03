# REACT 2026 starter

A local-first competition workspace. Task, target, metric, metric direction, and validation remain unset until launch. Read **AGENTS.md**, **COMPETITION.md**, and **CURRENT_STATE.md** before working.

## Install and verify

Run from this directory. PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
```

If script activation is blocked, use `.\.venv\Scripts\python.exe` directly. Linux/Kaggle uses `python` and, locally, `source .venv/bin/activate`. Optional models need their own explicit install: `pip install catboost`, `pip install lightgbm`, or `pip install xgboost`. Optional table readers: `pyarrow`, `openpyxl`, or `xlrd`. They are not required by the starter checks. No external datasets or pretrained weights are downloaded by these tools.

Smoke tests generate tiny fixtures in temporary directories and use SMOKE IDs outside the competition tracker. They verify plumbing, not model quality. See reports/starter_verification.md for what was actually tested.

## Inspect organizer files

Place only organizer-supplied files in `data/raw/` after launch:

```powershell
python src/audit.py
python src/audit.py --train data/raw/train.csv --test data/raw/test.csv --sample data/raw/sample_submission.csv
```

The report is `reports/data_audit.md`; older audits are preserved. Defaults scan CSV/TSV/JSONL and retain up to 100,000 deterministic sampled rows for diagnostics. Other table formats may be loaded fully before sampling. Use `--full` only when practical. Sample-based duplicate counts cannot establish absence. Ambiguous target/file roles remain unresolved. Audit never trains, transfers test labels, selects CV, or changes features.

## Configure after launch

Read the full statement and Evaluation first. Inspect the sample, verify the official metric with its worked example, investigate leakage and time/entity/group structure, then agree on validation.

```powershell
python src/config.py | Out-File -Encoding utf8 config.json
```

Edit that JSON (or `DEFAULT_CONFIG` in `src/config.py`). Set:

- `train_file`, `target`, `features`, `task` (`regression` or `classification`), and ID columns. Paths are relative to the project root unless absolute.
- `metric`, `metric_direction` (`higher`/`lower`), and any official metric parameters. Built-in names: `mae`, `mse`, `rmse`, `r2`, `accuracy`, `f1`, `precision`, `recall`, `log_loss`, `roc_auc`, `average_precision`. They are helpers, not the official definition.
- `prediction_kind`: `value`, `probability`, `label`, or `decision`. Classification requires explicit `class_order`; binary ranking/label conversion also needs `positive_class`. Binary probability-to-label conversion requires `label_threshold`. Multiclass probability columns follow `class_order`; argmax chooses the first class in a tie.
- `validation_type`: `kfold`, `stratified`, `group`, `stratified_group`, or `time`; also `validation_rationale`, `n_splits`, and `shuffle`. Group methods require `group_column`; time requires `time_column`, no shuffle, and optional `time_gap` in rows. Time CV sorts stably, rejects tied boundary timestamps, and reports irregular spacing. Initial unvalidated rows keep fold -1 and missing OOF predictions.
- `model`, `model_params`, and explicit seed (starter default 42). Model names: `linear`, `logistic`, `random_forest`, `extra_trees`, `catboost`, `lightgbm`, `xgboost`, `tfidf_logistic`, `tfidf_linear`. For TF-IDF set `features` to `[text_column]`. The tabular starter uses fold-local imputation, scaling, and one-hot encoding even for optional boosters; native categorical handling is a future controlled experiment.

For a custom official metric, add `src/official_metric.py` with `score(y_true, predictions, **params) -> float`, set `metric="custom"`, `custom_metric="src.official_metric:score"`, `custom_metric_kind`, and direction. The function receives original training labels and configured prediction outputs. Test it against the official example. No runtime guess selects a metric or averaging convention.

IDs are reserved for alignment and excluded from baseline features. Before experimenting with suspicious identifiers, document the decision and make the smallest reviewed feature-code change. No audit heuristic makes that decision.

## Run and inspect an experiment

Only after configuration is established:

```powershell
python src/train.py --config config.json --model catboost --experiment E001 --seed 42 --hypothesis "Document the question being tested" --change "Document the single controlled change" --no-test-inference
```

Every meaningful experiment has a new E ID, including failed runs. Set `splits_file` to a prior `outputs/reports/E001/splits.json` for identical folds; reuse rejects changed inputs or settings. Inputs are fingerprinted by exact file bytes, so reserialized files count as changed.

- `experiments/experiments.csv`: compact ledger; enter conclusions, notes, leaderboard scores, and exact notebook-version URLs manually.
- `outputs/reports/E001.json`: configuration, CV, status, timings, artifact links, source hashes, Git provenance.
- `outputs/reports/E001/`: saved source, exact splits, environment, complete fitted estimator parameters, and log.
- `outputs/oof/E001.csv` plus metadata: original row positions, configured IDs, fold assignment, and prediction columns. CV mean/std use fold scores; standard deviation uses ddof=0. Time CV explicitly reports partial coverage.
- `outputs/models/E001/`: saved fold pipelines. Load only trusted artifacts generated by this project.

The ledger is for one experiment process at a time. Brief locking protects edits; do not run concurrent training commands or edit the tracker while a command is updating it. A crashed stale lock requires confirming no process is active before removing that specific lock. No model is automatically declared best.

## Predict and generate a submission

Test inference is a separate stage after all model fitting:

```powershell
python src/predict.py --experiment E001 --test-file data/raw/test.csv --aggregation mean
```

Alternatively, explicitly supply `--predict-test --aggregation mean` to training and configure `test_file`. Disabled inference does not open test data. Fold inference averages numeric regression outputs, class probabilities, or decision scores. Label-only models cannot use this averaging path; use probabilities when appropriate. Early stopping is not configured by this starter.

Set `sample_file`, `submission_columns`, `submission_kind`, and `submission_alignment` from Evaluation. For IDs, configure their columns and use `id` alignment; IDs are read as strings to preserve leading zeros. Without IDs, explicitly choose `position` and verify the sample follows test order. A single binary probability column requires `positive_class`. Multiple probability columns must follow the explicit `class_order`. The starter rejects all missing predictions; any official exception requires a deliberate code change after launch.

```powershell
python src/submission.py --experiment E001 --config config.json --description baseline
```

This produces a descriptive `sub_E001_..._cv....csv` and a provenance sidecar. Generated CSVs use UTF-8 and LF line endings for Windows/Linux portability. The optional config overrides only submission fields, not training settings. Validation checks schema/order, row count, IDs, prediction type, finiteness, probability bounds and sums. It never repairs predictions or overwrites a file. **No command uploads to Kaggle.** A generated CSV does not consume a slot; record actual upload counts and scores manually.

## Blend explicitly

```powershell
python src/ensemble.py --experiment E003 --members E001 E002 --weights 0.6 0.4 --hypothesis "Compare complementary OOF errors" --change "Blend the two saved candidates" --predict-test
```

Provide distinct IDs and nonnegative weights summing to one. Arithmetic mean uses `--method mean` and explicit equal weights. Members must have identical data identities, folds, coverage, metric semantics, and class mapping. Test blending is opt-in and requires member predictions. OOF scores are recomputed using organizer training labels. Rank averaging and public-LB weight tuning are not implemented.

## Reproduce on Kaggle

```powershell
python src/export_notebook.py --experiment E003
```

The uniquely named notebook embeds the immutable saved source for the selected experiment and all ensemble ancestors, plus configurations, exact folds, hashes, and environment versions. It contains no competition data. The exported notebook runs each saved source version in an isolated Python process; it never imports the changing local checkout.

Upload/open that notebook manually, map organizer input paths, inspect recorded package versions, and run it in a fresh runtime and output directory. It retrains, predicts, and regenerates any recorded submissions. Source, resolved configuration, folds, input identity, and metadata remain exact checks. OOF, test, and submission predictions require identical shape/identity/class mapping and numerical agreement at recorded tolerances (`rtol=1e-7`, `atol=1e-9`); their byte hashes remain provenance diagnostics. The replay report records maximum absolute/relative differences, values outside tolerance, and the recomputed OOF metric. For final candidates, Kaggle is the canonical reproduction environment.

Before a serious candidate, commit code locally and ensure Git provenance is recorded. The starter initializes Git but does not create a remote or publish. Save the verified **exact committed Kaggle Notebook version**, and enter its version URL in the ledger, candidate report, and submission sidecar. Share privately with organizer accounts as instructed. CSV upload and up-to-two final selections always require human action.

## Image tasks: activate only if revealed

After inspecting organizer file layout, define a PyTorch Dataset from organizer paths/labels and fold indices; use separate train/validation DataLoaders. Load the exact disclosed public backbone directly in the notebook. Apply stochastic augmentation only to training data; validate deterministic preprocessing and seed workers. Train independently per approved fold, retain checkpoints and OOF by original IDs, then infer test images only after fitting. Decide preprocessing, loss, metric, and inference averaging together. No image framework, transformer system, weights, or image datasets are installed pre-launch.

## Highest-priority preparation

Open `reports/synthetic_smoke_rehearsal.ipynb` in a **private Kaggle runtime**, Run All, then Save Version. It embeds only generated fixture data and frozen source; it needs no competition files or submissions. Confirm phone-verified account access, dependencies, and exact reproduction before September 6. At launch, stop setup and follow AGENTS.md's sequence; form the competition Team before the first submission.

To regenerate the rehearsal from the current source into a new filename, set `REACT_SMOKE_NOTEBOOK` to that destination before running the tests. The destination must not already exist; normal tests do not write to this repository.
