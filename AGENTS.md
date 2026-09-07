# Permanent operating instructions

This is a short university Kaggle competition. Keep code fast, readable, reproducible, and easy to modify. The two team members work together; do not divide responsibilities between them. Never silently make competition-strategy decisions. Present evidence and alternatives when a decision is ambiguous.

## Required reading at the start of every Codex session

Read, in order: `COMPETITION.md`, `CURRENT_STATE.md` (and `reports/CURRENT_STATE.md` when report context is needed), `reports/APPROACHES.md`, `reports/LEARNINGS.md`, `experiments/experiments.csv`, then the relevant recent report and source files. Do not restart the competition reasoning from zero or reinterpret immutable experiments without evidence.

## Non-negotiable rules

1. Never use test labels or leaked information.
2. Never attempt to infer hidden test labels through probing or de-anonymization.
3. Inspect test data only for legitimate schema/distribution analysis and inference.
4. Never include test data in model fitting, target encoding, supervised feature selection, early stopping, or validation. Do not train a train-vs-test classifier under these rules.
5. Prevent preprocessing leakage: fit anything learned from data inside each training fold when necessary. This includes imputers, vocabulary, encoders, scaling, feature selection, and calibration.
6. Always investigate group, temporal, duplicate, ID, entity, and preprocessing leakage.
7. Never install, download, or incorporate external datasets unless competition rules later explicitly permit them. Organizer-provided extra unlabeled data still needs a rule-compliant, explicitly agreed use.
8. Public pretrained models are allowed. Record their source, exact model name/revision, and load them directly inside the submitted notebook. Disclose them in the method summary. Do not fine-tune on external data.
9. Never overwrite a previous submission.
10. Never overwrite important OOF predictions or experiment artifacts. Keep original source snapshots and saved splits immutable. Updating run status or adding manually supplied leaderboard/version metadata is permitted.
11. Every meaningful experiment gets a unique ID such as E001. Never reuse an ID, even after a failure. SMOKE IDs are for isolated synthetic checks only.
12. Every experiment records hypothesis, changes from previous/baseline, validation method, fold scores, mean CV, CV standard deviation, important parameters, random seed, training time when practical, output paths, conclusion, and public leaderboard score if later supplied manually. Conclusions and unknown scores remain empty until supplied; do not fabricate them.
13. Use explicit random seeds; record parameters and runtime versions. Do not promise bitwise determinism without checking it.
14. Do not silently change validation strategy. Document the decision and justification separately from confirmed rules.
15. Do not judge models by training performance.
16. Do not make multiple major unrelated changes in one experiment unless explicitly instructed.
17. Preserve reproducibility, including input fingerprints, exact folds, source snapshots, configuration, Git commit/dirty state, and eventually the exact committed Kaggle Notebook version.
18. Prefer readable modular code over notebook spaghetti. Avoid an enterprise ML platform, unnecessary packages, and blind search.
19. Do not chase tiny public leaderboard gains when CV disagrees.
20. Flag suspicious structure before exploiting it. Audit findings never automatically drop or promote a feature.
21. Never automatically submit to Kaggle. This repository must contain no automatic upload path.
22. Never spend a Kaggle submission without explicit human instruction. The team limit is five per day; verify the actual reset boundary rather than assuming local midnight.
23. Never select Kaggle final submissions automatically. Up to two selections are human decisions.
24. Final scoring candidates must map to exact experiment/code versions: CSV -> experiment -> configuration -> CV -> OOF -> source -> Git commit -> exact committed Kaggle Notebook version.

## Launch-day sequence

When the problem statement and competition files arrive, stop generic setup work:

1. Read the complete problem statement and official Evaluation definition/worked example.
2. Inspect sample_submission; run the audit.
3. Establish task, target, IDs, and submission semantics from evidence.
4. Implement and verify the official metric exactly; Evaluation overrides helper assumptions.
5. Investigate leakage and time/groups/entities/users/machines/locations/patients/sessions/duplicates.
6. Recommend chronological holdout or expanding-window validation for the live fraud task, with justification; do not accept shuffled random K-fold as the default.
7. Run the first baseline to verify the complete pipeline. Only then prioritize experiments by expected information gain and leaderboard value.

## Experiment, leaderboard, and final-selection policy

Follow QUESTION -> HYPOTHESIS -> CONTROLLED CHANGE -> CV RESULT -> INTERPRETATION -> KEEP / REJECT / INVESTIGATE. Use OOF evidence for ensembles; never automatically optimize weights against public scores.

Record LOCAL CV and manually supplied PUBLIC LB for every submitted model. Significant disagreement calls for investigation of validation mismatch, drift, group/time leakage, unstable metrics, overfitting, or a lucky public subset. A submission should answer a useful question or represent a serious candidate, not merely be slightly different. Public 60% and private 40% are separate subsets; qualification uses the private leaderboard.

Near the deadline, compare candidates using CV, variance, robustness, validation trustworthiness, diversity, public score, and private-score stability. Do not automatically pick the top two public scores. Every candidate must reproduce from the exact committed notebook version. Share privately with the organizers as required; no cross-team sharing or public disclosure of the approach before permitted.

## Working and verification

For every live REACT experiment, invoke `python -m src.react_runner supervise ...` rather than `run` directly. The supervisor launches the immutable runner unchanged, emits a 60-second terminal heartbeat, and reports the final artifact status; never poll PowerShell process IDs manually. After launching a post-fit report command, use `python -m src.react_runner wait-for <repository-relative-artifact-path>` rather than manual sleep/path checks.

Do not invent a task, target, metric, direction, or validation strategy. Keep assumptions separate from confirmed facts. Update CURRENT_STATE.md as verified facts arrive, without silently declaring a best model or changing counters.

After each meaningful stage, run proportionate checks, report what was created and any assumptions, identify remaining human decisions, and state the next highest-value action. Do not train on dummy/external data to develop competition models. The user explicitly permits a tiny locally generated synthetic integration fit and replay solely for smoke testing, in temporary directories outside the competition tracker.

No automatic uploads, account actions, final selections, external datasets, or pretrained downloads are part of starter verification.

Live experiment IDs use `R001`, `R002`, ... . Preserve mock `E001`–`E006` and synthetic `SMOKE###` history unchanged.
