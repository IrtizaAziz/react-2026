# E001 OOF error summary

Saved OOF predictions: `outputs/oof/E001.csv`. Labels were derived from the configured binary probability threshold of 0.5. This is descriptive analysis only; no further model was fitted.

## Overall

- OOF rows: 891 / 891 (100% coverage)
- Accuracy: 0.793516
- Confusion matrix (true rows 0, 1; predicted columns 0, 1): `[[465, 84], [100, 242]]`
- False negatives: 100; false positives: 84

## Accuracy slices

| Slice | N | Accuracy |
|---|---:|---:|
| Sex: female | 314 | 0.773885 |
| Sex: male | 577 | 0.804159 |
| Pclass: 1 | 216 | 0.754630 |
| Pclass: 2 | 184 | 0.875000 |
| Pclass: 3 | 491 | 0.780041 |
| Age: 0--15 | 83 | 0.626506 |
| Age: 16--39 | 468 | 0.799145 |
| Age: 40+ | 163 | 0.840491 |
| Age missing | 177 | 0.813559 |

The clearest weak slice is passengers aged 0--15. This does not justify a feature change or model tuning by itself; future changes require a separately approved experiment.
