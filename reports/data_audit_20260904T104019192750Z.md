# Data audit

Generated: 2026-09-03T13:05:26.560761+00:00

Diagnostics only. No features are removed, no model is fitted, and no test labels are inferred.
Default sample cap: 100000; mode: bounded deterministic sample; seed: 42.
Counts and distributions below refer to analyzed rows unless explicitly called exact file totals.
JSON, Parquet, and Excel readers may load a complete table before sampling; CSV/TSV/JSONL use bounded samples.

## File inventory
No competition files found. Place organizer-provided files in data/raw/ after launch.

## Roles, target, and train/test diagnostics
train_file: unavailable; configure its path explicitly.
test_file: unavailable; configure its path explicitly.
sample_file: unavailable; configure its path explicitly.
Target uncertain or unavailable. Target-dependent diagnostics were not performed.

## POTENTIAL LEAKAGE RISKS
Heuristics and drift signals are diagnostics, not proof. Review before using or removing features.
### CRITICAL
- No signal found by these checks; this does not establish absence of leakage.
### HIGH
- No signal found by these checks; this does not establish absence of leakage.
### MEDIUM
- No signal found by these checks; this does not establish absence of leakage.
### LOW
- No signal found by these checks; this does not establish absence of leakage.

Still requires human review: collection process, post-outcome features, time/groups/entities, duplicate policies, and fold-local preprocessing.
