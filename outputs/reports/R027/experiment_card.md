# R027 experiment card

Derived factual companion to `decision_report.json`; feature importances are descriptive, not causal.

## Experiment

- Experiment ID: R027
- Parent: R026
- Incumbent: R017
- Timestamp: 2026-09-07T10:17:13.381774+00:00
- Immutable status: completed
- Hypothesis: The device-location share of device history captures incremental device-context familiarity beyond R026, while the count, novelty, and recency features bundled in R024 may have diluted the useful relationship-share signal.
- Feature module: src.device_location
- Ordered added features: device_location_share_of_device_history
- Feature count before/after: 51.000000 / 52.000000
- Inherited recipe: model=catboost; parameters={'loss_function': 'Logloss', 'iterations': 800, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'thread_count': 8, 'task_type': 'CPU', 'one_hot_max_size': 64}; parent config hash=cc92c23eea4ed65e1dc962f266002bf8ac5ea8ba90f1037165740e8f7d4cddc3
- Documented coherent change: append exactly the listed feature(s); no other generic-spec change is recorded.

## Metrics

| Metric | Score | Δ parent | Δ incumbent |
|---|---:|---:|---:|
| F1 | 0.797359 | 0.002677 | 0.007530 |
| F2 | 0.731707 | -0.000083 | 0.002112 |
| F2 early | 0.790518 | -0.000048 | 0.001574 |
| corrected F2 late | 0.678816 | -0.000067 | 0.002292 |
| July | 0.511018 | 0.000343 | 0.001737 |
| June 15–30 | 0.812410 | -0.000211 | 0.002277 |
| July 1–7 | 0.475920 | 0.001838 | 0.003715 |
| July 8–15 | 0.537728 | -0.000496 | 0.000672 |

## Decision

- Classification: FLAT-MIXED
- Submission gates passed: 0.000000
- Gate evidence: July_1_15=False, F2_late_corrected=True, F2=False, F1=True, July_1_7=True, July_8_15=True
- Early stopped: 0.000000
- Early-stop policy: {'F2': 0.01, 'F2_late_corrected': 0.005, 'July_1_15': 0.005}

## Feature importances

- Added features, F1: device_location_share_of_device_history (3.371074)
- Added features, F2: device_location_share_of_device_history (3.695672)
- Top 10 overall, F1: customer_location_count_share (9.279696), customer_log_amount_minus_prior_mean (7.532977), hour (5.549457), log_amount_bdt (5.399542), customer_device_count_share (4.839467), location (4.571645), merchant_new_customer_share_24h (3.606496), device_location_share_of_device_history (3.371074), amount_bdt (3.170603), customer_seconds_since_last (2.931229)
- Top 10 overall, F2: customer_log_amount_minus_prior_mean (9.966463), customer_location_count_share (8.159606), merchant_new_customer_share_24h (6.506716), customer_device_count_share (4.947374), location (4.190594), device_location_share_of_device_history (3.695672), customer_device_prior_count (3.566412), customer_amount_to_prior_mean (3.276295), hour (3.222761), customer_share_of_device_prior_transactions (3.116267)

## Certification and provenance

- Certification status: new
- Certification results: {'strict_past': True, 'oracle': True, 'equal_timestamp_isolation': True, 'permutation_invariance': True, 'duplicate_same_timestamp_pair': True, 'future_independence': True, 'label_independence': True, 'chunk_vs_whole': True, 'missing_location_sentinel': True}
- Parent matrix parity / row-ID alignment: verified by pre-fit candidate checks
- Locked split signature: 6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47
- Train input fingerprint: {'sha256': '06823ea80cceade018b2428e0c7c9cb89e4efae45fcad8cbf61e8b27b350df20', 'rows': 731942, 'columns': ['transaction_id', 'customer_id', 'timestamp', 'amount_bdt', 'merchant_id', 'merchant_category', 'device_id', 'device_type', 'location', 'payment_method', 'transaction_type', 'account_age_days', 'fraud']}
- Test inference / submission: False / False
- Runtime seconds: 58.849976
- OOF: outputs/oof/R027.csv
- Model/config/spec/report/source: ['outputs/models/R027/fold_1.pkl', 'outputs/models/R027/fold_0.pkl']; outputs/reports/R027//config.json; specs\r027.json; outputs/reports/R027//decision_report.json; outputs/reports/R027//source
- Module hashes: {'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_history.py': '1e43ea1b15c7b86c4187917bdc35fe40f30ffbdd4695458d4f2125ae82b7c779', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_relationships.py': 'c83515d9c37621532aae8fc321b2c55bcf4271663f808a71af2b1090ad4ecf31', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\device_location.py': 'ec49b6abc31f85e98d17ecd5715075e27566c164aaf162494a4af8d6893d9997'}
- Runner hashes: {'feature_experiment.py': '9289382135ac0275d7463c8c9b84d3c89fe16b51dc1624e4aa6191817af15086', 'react_runner.py': 'b24854c977f37d9e2b3466639f5d07e6e968184bffc13b3577b6d9cc4ca05a5f'}
- Parent cache: {'used': True, 'path': 'outputs\\feature_matrices\\R026.pkl', 'matrix_sha256': '54881bb19b4d9c306fc3731247a8b9ba3dc10170541d2e86b3f8018dbf454887', 'manifest_sha256': '362290205af1e8bc3dd5068fff11c029c68f5b8f017ed95a8554357f06397282'}

## Reference-run analysis

- R024: compatible=False; deltas=UNKNOWN

## Factual conclusion

R027 improved F1, July, July 1–7 versus its parent; classification is FLAT-MIXED and submission gates passed=0.000000.
