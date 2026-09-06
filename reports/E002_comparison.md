# E002 controlled comparison with E001

E002 reused E001's saved split signature exactly. Its only change is row-local `FamilySize = SibSp + Parch + 1` and `IsAlone = (FamilySize == 1)`.

## Overall

| Experiment | Fold scores | Mean accuracy | Population std | OOF coverage |
|---|---|---:|---:|---:|
| E001 | 0.770950, 0.803371, 0.792135, 0.780899, 0.820225 | 0.793516 | 0.017209 | 100% |
| E002 | 0.776536, 0.797753, 0.803371, 0.797753, 0.831461 | 0.801375 | 0.017620 | 100% |

Absolute CV change: +0.007859.

All 891 OOF probability estimates changed. At the configured 0.5 threshold, 25 of 891 predicted labels changed.

| Experiment | Confusion matrix (true rows 0, 1; predicted columns 0, 1) |
|---|---|
| E001 | `[[465, 84], [100, 242]]` |
| E002 | `[[471, 78], [99, 243]]` |

## Accuracy slices

| Slice | N | E001 | E002 | Change |
|---|---:|---:|---:|---:|
| Age 0--15 | 83 | 0.626506 | 0.662651 | +0.036145 |
| Passenger class 1 | 216 | 0.754630 | 0.763889 | +0.009259 |
| Passenger class 2 | 184 | 0.875000 | 0.875000 | +0.000000 |
| Passenger class 3 | 491 | 0.780041 | 0.790224 | +0.010183 |
| Alone | 537 | 0.819367 | 0.824953 | +0.005587 |
| Not alone | 354 | 0.754237 | 0.765537 | +0.011299 |

The improvement is directionally consistent with the hypothesis, especially for children and not-alone passengers. It is a single controlled five-fold comparison, so it is evidence for further review rather than a conclusion to tune or expand features automatically.
