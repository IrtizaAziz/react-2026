# Current state — live Stage 1

The live competition is binary fraud probability prediction scored by `sklearn.metrics.average_precision_score` (higher is better). The supplied files are resolved from the verified workspace root via `config_live_stage1.json`.

Stage 1 locks the static feature profile and two `calendar_time` folds:

- F1: train `< 2026-03-14`; validate `[2026-03-14, 2026-05-15)`.
- F2: train `< 2026-05-15`; validate `[2026-05-15, 2026-07-16)`.

`reports/live_foundation.json` and `outputs/reports/live_foundation_splits.json` are the source of truth for the resulting file hashes, split signature, and exact counts. Stage 1 does not train a model, reserve an R ID, make predictions, create a submission, or contact Kaggle.

The E001–E006 Titanic rehearsal, its scores, and any mock counters remain preserved historical artifacts. They are not comparable to live Average Precision runs. Live work starts at R001 after the Stage 1 review.

R021 is an immutable, OOF-only fixed probability blend of R017 and R020 at 0.75/0.25. It has no test predictions or submission artifact. Relative to R017, it improves F1 by 0.00541234, F2 by 0.00224244, and July by 0.00212758, but does not satisfy the operational scarce-submission gates (July < roughly +0.003, corrected F2 late +0.00186691 < roughly +0.002, and F2 +0.00224244 < roughly +0.003). Its classification is PROMISING-BUT-BELOW-SUBMISSION-GATE and R017 remains the primary incumbent.

R025 is an immutable R017-derived merchant-side new-customer relationship-composition test. It is FLAT-MIXED: F2 +0.001336, corrected F2 late +0.000966, July +0.000576, and F1 +0.008144 versus R017. It does not clear the fixed July, corrected-late, or F2 scarce-submission gates; no test predictions or submission artifact exist.
