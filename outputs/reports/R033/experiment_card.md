# R033 experiment card

Derived factual companion to `decision_report.json`; feature importances are descriptive, not causal.

## Experiment

- Experiment ID: R033
- Parent: R029
- Incumbent: R017
- Timestamp: 2026-09-07T11:21:59.657151+00:00
- Immutable status: completed
- Hypothesis: Acute cross-customer device sharing provides incremental fraud-ranking signal beyond R029's merchant customer-composition features, so combining the independently useful device-sharing signal from R032 with the promoted merchant-composition lineage may produce additive recent-period gains.
- Feature module: src.device_recent_sharing
- Ordered added features: device_other_customers_24h
- Feature count before/after: 52.000000 / 53.000000
- Inherited recipe: model=catboost; parameters={'loss_function': 'Logloss', 'iterations': 800, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'thread_count': 8, 'task_type': 'CPU', 'one_hot_max_size': 64}; parent config hash=35ce87bb3bc58be5cff23cea9418b7a954322563c1015f5e0727c1bb04fc3545
- Documented coherent change: append exactly the listed feature(s); no other generic-spec change is recorded.

## Metrics

| Metric | Score | Delta parent | Delta incumbent |
|---|---:|---:|---:|
| F1 | 0.795969 | -0.001603 | 0.006140 |
| F2 | 0.733016 | 0.000540 | 0.003422 |
| F2 early | 0.792950 | 0.001696 | 0.004006 |
| corrected F2 late | 0.678836 | -0.000690 | 0.002311 |
| July | 0.511443 | -0.000739 | 0.002162 |
| June 15-30 | 0.812118 | -0.000755 | 0.001984 |
| July 1-7 | 0.476344 | -0.001484 | 0.004140 |
| July 8-15 | 0.538261 | 0.000064 | 0.001204 |

## Decision

- Classification: FLAT-MIXED
- Submission gates passed: false
- Gate evidence: July_1_15=false, F2_late_corrected=true, F2=true, F1=true, July_1_7=true, July_8_15=true
- Early stopped: false
- Early-stop policy: {'F2': 0.01, 'F2_late_corrected': 0.005, 'July_1_15': 0.005}

## Feature importances

- Added features, F1: device_other_customers_24h (0.553519)
- Added features, F2: device_other_customers_24h (1.161533)
- Top 10 overall, F1: customer_location_count_share (9.586607), customer_log_amount_minus_prior_mean (6.236378), customer_location_prior_count (6.058630), hour (5.583123), location (5.232771), customer_device_count_share (5.100865), log_amount_bdt (4.409805), amount_bdt (3.636654), customer_amount_to_prior_mean (3.272702), customer_device_prior_count (3.011403)
- Top 10 overall, F2: customer_log_amount_minus_prior_mean (11.371574), customer_location_count_share (10.581199), location (5.299275), merchant_new_customer_share_24h (5.190880), customer_device_count_share (5.132885), customer_share_of_device_prior_transactions (3.582617), customer_location_prior_count (3.296009), hour (3.290018), customer_seconds_since_last (3.038780), log_amount_bdt (3.009598)

## Certification and provenance

- Certification status: reused
- Certification results: {'strict_past': True, 'equal_timestamp_isolation': True, 'oracle': True, 'permutation_invariance': True, 'duplicate_same_timestamp_pair': True, 'future_independence': True, 'label_independence': True, 'chunk_vs_whole': True, 'excluding_current_customer': True, 'raw_frame_compatibility': True}
- Parent matrix parity / row-ID alignment: verified by pre-fit candidate checks
- Locked split signature: 6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47
- Train input fingerprint: {'sha256': '06823ea80cceade018b2428e0c7c9cb89e4efae45fcad8cbf61e8b27b350df20', 'rows': 731942, 'columns': ['transaction_id', 'customer_id', 'timestamp', 'amount_bdt', 'merchant_id', 'merchant_category', 'device_id', 'device_type', 'location', 'payment_method', 'transaction_type', 'account_age_days', 'fraud']}
- Test inference / submission: False / False
- Runtime seconds: 58.020079
- OOF: outputs/oof/R033.csv
- Model/config/spec/report/source: ['outputs/models/R033/fold_1.pkl', 'outputs/models/R033/fold_0.pkl']; outputs/reports/R033//config.json; specs\r033.json; outputs/reports/R033//decision_report.json; outputs/reports/R033//source
- Module hashes: {'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_history.py': '1e43ea1b15c7b86c4187917bdc35fe40f30ffbdd4695458d4f2125ae82b7c779', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_relationships.py': 'c83515d9c37621532aae8fc321b2c55bcf4271663f808a71af2b1090ad4ecf31', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\device_recent_sharing.py': '00933ba3078c84aba8db39a45b728a5ecffaa81aecda78673a7dfcfa458f2836'}
- Runner hashes: {'feature_experiment.py': '1c550f26933f32c9f585d12095b5c4f4f6b789c15a43495c64c937042b61b5c5', 'react_runner.py': '662ca45b1121307c5160726415d3fcd4a654109328c0d7261a445a5c284bc6e0'}
- Parent cache: {'used': True, 'path': 'outputs\\feature_matrices\\R029.pkl', 'matrix_sha256': 'ab767d22d77a44ef3514e8580d9287be720b210475037158514883c1c249771c', 'manifest_sha256': '433acf180bbe536d3d86d2a2cf559ae2ca4368548cb887ae56a601dbe02a028a'}

## Reference-run analysis

- R032: compatible=True; deltas={'F2': 0.00013515342690861853, 'F1': 0.000593509736459108, 'F2_early': 0.0007815749138812, 'F2_late_corrected': -0.0002991474507759717}
- R026: compatible=True; deltas={'F2': 0.0012261505159312236, 'F1': 0.0012868752800776484, 'F2_early': 0.002384352895868802, 'F2_late_corrected': -4.754140829665854e-05}

## Factual conclusion

R033 improved F2, F2 early, July 8-15 versus its parent; classification is FLAT-MIXED and submission gates passed=false.
