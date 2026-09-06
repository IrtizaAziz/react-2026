# E006 raw Title comparison with E005

E006 reused E005's exact saved split signature, CatBoost settings, and features. Its only change is the deterministic, row-local raw `Title` string between the comma and period in `Name`, passed as a native categorical feature. No surname or group identity was derived.

## Overall

| Experiment | Fold scores | Mean accuracy | Population std | OOF coverage |
|---|---|---:|---:|---:|
| E005 | 0.837989, 0.848315, 0.797753, 0.831461, 0.842697 | 0.831643 | 0.017827 | 100% |
| E006 | 0.832402, 0.825843, 0.820225, 0.837079, 0.825843 | 0.828278 | 0.005851 | 100% |

Absolute CV change: -0.003365. Two of five fold scores improved. The positive-class OOF probability correlation is 0.983976; 37 of 891 thresholded OOF labels changed.

| Experiment | Confusion matrix (true rows 0, 1; predicted columns 0, 1) |
|---|---|
| E005 | `[[507, 42], [108, 234]]` |
| E006 | `[[498, 51], [102, 240]]` |

E005 correct / E006 wrong: 20 rows. E006 correct / E005 wrong: 17 rows.

## Slices

Age 0--15: E005 0.855422 (N=83); E006 0.891566; change +0.036145.

Title groups with at least 10 training rows:

| Title | N | E005 | E006 | Change |
|---|---:|---:|---:|---:|
| Master | 40 | 0.850000 | 0.950000 | +0.100000 |
| Miss | 182 | 0.818681 | 0.813187 | -0.005495 |
| Mr | 517 | 0.837524 | 0.839458 | +0.001934 |
| Mrs | 125 | 0.824000 | 0.808000 | -0.016000 |

## Decision: REJECT

Raw Title improves the children and Master slices but reduces overall paired CV, improves only two folds, and has 20 newly wrong rows versus 17 newly correct rows. Keep E005 as the current candidate; do not add raw Title to it.
