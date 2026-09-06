# Gap analysis — REACT 2026 preparation

## Already present

The starter already had a descriptive audit, explicit task/metric configuration, KFold/stratified/group/time splitters with persisted assignments, fold-local preprocessing, native CatBoost handling, OOF and test artifacts, immutable experiment IDs, submission schema/order validation, source/config/environment snapshots, and a self-contained notebook exporter. E001–E006 and their failed runs are preserved.

## Missing

There was no operator-facing experiment ranking command, no matched-fold screening helper, no explicit ranking blend guardrail, no report/research runbook set, no split core/optional dependency files, and no isolated AutoGluon challenger entry point.

## Improve

Extend the current ensemble module rather than creating a second artifact format; use the existing experiment ledger and saved OOF metadata as the comparison source; keep optional frameworks outside the native pipeline.

## Avoid

Do not vendor large external repositories, install every framework into the core environment, tune weights against the public leaderboard, or let AutoML replace locked validation. MLEvolve remains explicitly out of scope.

## Implementation order

1. Protect the existing contracts and run the rehearsal tests.
2. Add comparison, OOF diagnostics, and matched-fold screening.
3. Add explicit, metric-gated ensemble diversity methods.
4. Add runbooks, research template, and dependency split.
5. Add optional challengers only behind clean imports and non-comparable validation labels.
6. Rehearse fresh-runtime Kaggle notebook replay before launch.
