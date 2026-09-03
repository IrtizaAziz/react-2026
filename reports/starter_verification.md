# Starter verification

Verified locally on 03 September 2026. No competition data was available or used.

## Ready

- Requested repository structure, permanent operating instructions, competition record, unchanged unknown state, empty experiment ledger, and three notebook templates.
- Descriptive audit with uncertainty, sample limits, duplicate/drift diagnostics, and ranked leakage findings. Initial empty-data report: `data_audit.md`.
- Explicit configuration gates, all five requested splitters, saved-split validation, metric helpers, fold-local preprocessing, lazy optional model adapters, and CV training/inference.
- Exclusive experiment/artifact IDs, failed-run preservation, source snapshots, environment records, saved folds, and model/prediction hashes.
- Compatible OOF blending, sample-based submission validation, and export of saved source for individual experiments or ensemble ancestors.
- Self-contained synthetic rehearsal: `synthetic_smoke_rehearsal.ipynb`. It contains only locally generated fixtures and must never be used as a competition submission.

## Checks run

`python -m unittest discover -s tests -v`: **11 tests passed**; final run took 20.386 seconds.

Coverage:

1. Imports and CLI help for all runnable entrypoints.
2. Unset competition configuration refuses training before reserving an ID.
3. Tiny synthetic classification pipeline, fold-local categorical vocabulary, saved pipeline reload, inference, explicit class mapping, sample-ID reordering, and empty human-entered tracker fields.
4. Experiment, prediction, and submission overwrite protection.
5. Injected pre-fit failure keeps its ID reserved and records failed status.
6. Hand-calculated RMSE, log loss, accuracy, and binary ROC AUC; incompatible directions/averaging and invalid probabilities rejected.
7. Duplicate/missing IDs, wrong submission columns, and ambiguous alignment rejected.
8. Deterministic saved split reuse, changed-data rejection, group isolation, chronological splits with uncovered OOF rows, tied-time rejection, and unavailable StratifiedGroupKFold handling.
9. Empty audit, uncertain target, sample-assisted target candidate, conflicting training duplicates, train/test overlap reporting, deterministic sampling, and audit-history preservation.
10. Optional booster dependency errors and incompatible LinearSVC probability requests.
11. Explicit two-member OOF blend, bad weights/class-order rejection, and full exported-notebook replay including training, inference, and two submission CSVs. The preserved rehearsal notebook was executed locally; prediction and submission hashes matched exactly.

Additional checks: all source files compiled; all modules imported; the three template notebooks' code cells compiled; the root empty-data audit ran; Git ignore rules correctly exclude raw/processed data, models, OOF/test predictions, submission CSVs, virtual environments, and credential files. The ledger and lightweight reports remain visible to Git. Generated CSVs use canonical LF line endings (asserted in tests), and .gitattributes preserves source line endings across Windows/Linux checkouts.

The integration fixture uses 12 generated training rows and 3 generated inference rows. A second member fixture copies the first run solely to exercise ensemble orchestration. Export replay refits those tiny fixtures. These scores are not competition results. Test artifacts stayed in temporary directories and the competition ledger contains only its header.

## Environment and limits

- Windows; project-local `.venv`; Python 3.14.4.
- NumPy 2.5.2; pandas 3.0.5; scikit-learn 1.9.0; matplotlib 3.11.1.
- CatBoost, LightGBM, XGBoost, optional Parquet/Excel readers, PyTorch, and pretrained models were not installed or fitted. Only the optional-dependency failure paths were tested.
- Kaggle account access, GPU availability, package compatibility, and Save Version were not exercised here. Replay checks are strict: a different runtime or nondeterministic model can fail exact hashes and requires investigation.
- Git was initialized without a commit, remote, or publication. Make a local source commit before a serious candidate; every run records whether a committed version is available.
- Deadline timezone, daily submission reset boundary, target, task, metric/direction, submission contract, and validation decisions still require launch evidence. No training, uploading, or final selection will resolve these automatically.

## Single highest-priority preparation step

Open `synthetic_smoke_rehearsal.ipynb` in a **private Kaggle Notebook**, Run All, and Save Version. This rehearses account/runtime access and reproducible training/inference without competition files or a submission slot.
