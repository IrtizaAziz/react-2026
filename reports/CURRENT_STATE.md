# Current state — live Stage 1

The live competition is binary fraud probability prediction scored by `sklearn.metrics.average_precision_score` (higher is better). The supplied files are resolved from the verified workspace root via `config_live_stage1.json`.

Stage 1 locks the static feature profile and two `calendar_time` folds:

- F1: train `< 2026-03-14`; validate `[2026-03-14, 2026-05-15)`.
- F2: train `< 2026-05-15`; validate `[2026-05-15, 2026-07-16)`.

`reports/live_foundation.json` and `outputs/reports/live_foundation_splits.json` are the source of truth for the resulting file hashes, split signature, and exact counts. Stage 1 does not train a model, reserve an R ID, make predictions, create a submission, or contact Kaggle.

The E001–E006 Titanic rehearsal, its scores, and any mock counters remain preserved historical artifacts. They are not comparable to live Average Precision runs. Live work starts at R001 after the Stage 1 review.
