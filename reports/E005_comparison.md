# E005 CatBoost comparison with E002

E005 replaces the intended CatBoost runs E003 and E004, both of which failed before fitting. Before E005 was reserved, an untracked one-fold preflight confirmed: zero train/validation row overlap; fold-local preprocessing; transformed native categorical indices `[5, 6, 7, 8]`; CatBoost fitting; and finite, row-normalized `(179, 2)` validation probabilities in class order 0, 1. It did not write an experiment, OOF, test, or submission artifact.

E005 reused E001's saved split signature exactly and used only E002's nine features.

## Overall

| Experiment | Fold scores | Mean accuracy | Population std | OOF coverage |
|---|---|---:|---:|---:|
| E002 | 0.776536, 0.797753, 0.803371, 0.797753, 0.831461 | 0.801375 | 0.017620 | 100% |
| E005 | 0.837989, 0.848315, 0.797753, 0.831461, 0.842697 | 0.831643 | 0.017827 | 100% |

Absolute CV change: +0.030268. Pearson correlation between E002 and E005 positive-class OOF probabilities: 0.919869. At the configured threshold of 0.5, 95 of 891 OOF labels changed.

| Experiment | Confusion matrix (true rows 0, 1; predicted columns 0, 1) |
|---|---|
| E002 | `[[471, 78], [99, 243]]` |
| E005 | `[[507, 42], [108, 234]]` |

E002 correct / E005 wrong: 34 rows. E005 correct / E002 wrong: 61 rows.

## Accuracy slices

| Slice | N | E002 | E005 | Change |
|---|---:|---:|---:|---:|
| Age 0--15 | 83 | 0.662651 | 0.855422 | +0.192771 |
| Passenger class 1 | 216 | 0.763889 | 0.773148 | +0.009259 |
| Passenger class 2 | 184 | 0.875000 | 0.923913 | +0.048913 |
| Passenger class 3 | 491 | 0.790224 | 0.822811 | +0.032587 |
| Alone | 537 | 0.824953 | 0.837989 | +0.013035 |
| Not alone | 354 | 0.765537 | 0.822034 | +0.056497 |

## Interpretation

E005 is stronger than E002 in this controlled local comparison: it has a substantial mean-CV gain and improves every requested slice, with especially large gains for passengers aged 0--15 and not travelling alone. Its 0.919869 probability correlation and 95 changed labels also leave meaningful prediction diversity, but the primary conclusion is stronger—not merely ensemble diversity. No blend or tuning was performed.
