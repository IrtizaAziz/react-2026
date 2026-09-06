## Final results

| Experiment | Status | Local CV accuracy | Public LB accuracy | CV - LB | Notes |
|---|---|---:|---:|---:|---|
| E001 | completed | 0.793516 | — | — | Logistic baseline |
| E002 | completed | 0.801375 | 0.763150 | +0.038225 | FamilySize + IsAlone |
| E003 | failed | — | — | — | Preserved failed CatBoost integration attempt |
| E004 | failed | — | — | — | Preserved failed CatBoost integration attempt |
| E005 | selected best mock experiment | 0.831643 | 0.775110 | +0.056533 | CatBoost replacement |
| E006 | rejected | 0.828278 | — | — | Raw Title reduced overall CV |

E005 - E002 improvement: local CV +0.030268; public LB +0.011960. Local CV overestimated the improvement magnitude by +0.018308.

## What worked

- Audit before modeling established task semantics, missingness, ID risks, entity overlap, and drift before any fit.
- Explicit validation fixed a reproducible five-fold stratified protocol for the non-entity comparisons.
- The controlled FamilySize/IsAlone experiment isolated a single feature change.
- The CatBoost model-family comparison used the same information and immutable folds as E002.
- Immutable experiment tracking preserved configurations, splits, source snapshots, artifacts, and failed E003/E004 records.
- The local-to-Kaggle submission workflow ultimately produced a validator-confirmed `PassengerId,Survived` CSV from saved predictions.

## Problems discovered

- Categorical imputer configuration bug: the starter used a constant missing token instead of requested most-frequent imputation. Fixed; the imputer now uses most-frequent strategy and the starter test suite passed.
- Derived-feature fingerprint issue: in-memory derived columns changed the raw-data fingerprint and blocked saved-split reuse. Fixed; provenance fingerprints the supplied raw table before deterministic derivation, and E002/E005/E006 reused the saved split signature.
- CatBoost sklearn cloning issue: CatBoost normalized `cat_features`, which sklearn clone rejected. Fixed; CatBoost uses an unfitted deep copy per fold while other models retain sklearn clone. The one-fold preflight and E005 completed successfully.
- CatBoost transformed categorical-index issue: sklearn preprocessing generated temporary column names, so source categorical names could not resolve in CatBoost. Fixed; stable transformed categorical indices are passed. The preflight and E005 completed successfully.
- Incorrect E002 submission packaging: the internal prediction artifact (`__row__, PassengerId, pred_0, pred_1`) was uploaded instead of a submission CSV. Fixed procedurally; a new descriptive E002 CSV was regenerated solely from saved predictions and passed the actual-template validator with 418 aligned integer labels.

## Validation lesson

Local CV correctly ranked E005 above E002: 0.831643 versus 0.801375. It overestimated the magnitude of the improvement: +0.030268 locally versus +0.011960 on the public leaderboard. Treat local CV as the primary ranking signal, but require public/private leaderboard comparisons to assess calibration of expected gains rather than optimizing to the public score.

## Launch-day implications

- Run the audit and lock task, metric, ID, submission, leakage, and validation decisions before the first experiment.
- Save and reuse immutable splits for every controlled comparison.
- Run a one-fold untracked integration preflight for any new model adapter or feature path before reserving an experiment ID.
- Keep preprocessing and feature derivation fold-local; fingerprint supplied raw inputs before deterministic in-memory transformations.
- Treat CatBoost native categorical column positions as part of the tested adapter contract.
- Upload only files from `submissions/`, never artifacts from `outputs/predictions/`; verify columns, row count, IDs, and labels against the organizer sample immediately before upload.
- Record every actual public score against the exact submission sidecar and compare CV/LB gaps before deciding whether further experimentation is justified.
