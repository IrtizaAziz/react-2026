# R034 experiment card

Derived factual companion to `decision_report.json`; feature importances are descriptive, not causal.

## Experiment

- Experiment ID: R034
- Parent: R029
- Incumbent: R017
- Timestamp: 2026-09-07T11:36:19.449329+00:00
- Immutable status: completed
- Hypothesis: A transaction that occurs at an unusual time relative to the customer's own strictly prior temporal habits provides incremental fraud-ranking signal beyond R029's behavioral, relationship, merchant, and composition features.
- Feature module: src.customer_temporal_habits
- Ordered added features: customer_daypart_share, customer_weekday_share
- Feature count before/after: 52.000000 / 54.000000
- Inherited recipe: model=catboost; parameters={'loss_function': 'Logloss', 'iterations': 800, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'thread_count': 8, 'task_type': 'CPU', 'one_hot_max_size': 64}; parent config hash=35ce87bb3bc58be5cff23cea9418b7a954322563c1015f5e0727c1bb04fc3545
- Documented coherent change: append exactly the listed feature(s); no other generic-spec change is recorded.

## Metrics

| Metric | Score | Delta parent | Delta incumbent |
|---|---:|---:|---:|
| F1 | 0.796896 | -0.000675 | 0.007067 |
| F2 | 0.732131 | -0.000344 | 0.002537 |
| F2 early | 0.791079 | -0.000176 | 0.002134 |
| corrected F2 late | 0.678968 | -0.000558 | 0.002444 |
| July | 0.511513 | -0.000669 | 0.002232 |
| June 15-30 | 0.812307 | -0.000566 | 0.002174 |
| July 1-7 | 0.477825 | -0.000003 | 0.005620 |
| July 8-15 | 0.537089 | -0.001108 | 0.000032 |

## Decision

- Classification: FLAT-MIXED
- Submission gates passed: false
- Gate evidence: July_1_15=false, F2_late_corrected=true, F2=false, F1=true, July_1_7=true, July_8_15=true
- Early stopped: false
- Early-stop policy: {'F2': 0.01, 'F2_late_corrected': 0.005, 'July_1_15': 0.005}

## Feature importances

- Added features, F1: customer_daypart_share (0.979287), customer_weekday_share (0.507788)
- Added features, F2: customer_daypart_share (0.533635), customer_weekday_share (0.523397)
- Top 10 overall, F1: customer_location_count_share (10.370353), customer_log_amount_minus_prior_mean (5.853660), location (5.090710), customer_amount_to_prior_mean (4.810295), log_amount_bdt (4.648724), hour (4.359847), customer_device_count_share (4.309450), customer_location_prior_count (4.207018), merchant_new_customer_share_24h (3.599739), amount_bdt (3.494017)
- Top 10 overall, F2: customer_location_count_share (10.869990), customer_log_amount_minus_prior_mean (9.953683), merchant_new_customer_share_24h (5.850492), location (4.884961), customer_device_count_share (4.176568), hour (3.542290), device_prior_distinct_customer_count_excluding_current (3.197626), customer_device_prior_count (3.116778), customer_amount_to_prior_mean (2.925227), log_amount_bdt (2.922846)

## Certification and provenance

- Certification status: new
- Certification results: {'oracle': True, 'strict_past': True, 'equal_timestamp_isolation': True, 'permutation_invariance': True, 'duplicate_same_timestamp_pair': True, 'future_independence': True, 'label_independence': True, 'chunk_vs_whole': True, 'raw_frame_compatibility': True}
- Parent matrix parity / row-ID alignment: verified by pre-fit candidate checks
- Locked split signature: 6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47
- Train input fingerprint: {'sha256': '06823ea80cceade018b2428e0c7c9cb89e4efae45fcad8cbf61e8b27b350df20', 'rows': 731942, 'columns': ['transaction_id', 'customer_id', 'timestamp', 'amount_bdt', 'merchant_id', 'merchant_category', 'device_id', 'device_type', 'location', 'payment_method', 'transaction_type', 'account_age_days', 'fraud']}
- Test inference / submission: False / False
- Runtime seconds: 54.651692
- OOF: outputs/oof/R034.csv
- Model/config/spec/report/source: ['outputs/models/R034/fold_1.pkl', 'outputs/models/R034/fold_0.pkl']; outputs/reports/R034//config.json; specs\r034.json; outputs/reports/R034//decision_report.json; outputs/reports/R034//source
- Module hashes: {'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_temporal_habits.py': '50eb81f731a2a39fd38c3a569bde5c26244f77592f1ed9649d2e9877a7b56e61'}
- Runner hashes: {'feature_experiment.py': '1c550f26933f32c9f585d12095b5c4f4f6b789c15a43495c64c937042b61b5c5', 'react_runner.py': '662ca45b1121307c5160726415d3fcd4a654109328c0d7261a445a5c284bc6e0'}
- Parent cache: {'used': True, 'path': 'outputs\\feature_matrices\\R029.pkl', 'matrix_sha256': 'ab767d22d77a44ef3514e8580d9287be720b210475037158514883c1c249771c', 'manifest_sha256': '433acf180bbe536d3d86d2a2cf559ae2ca4368548cb887ae56a601dbe02a028a'}

## Factual conclusion

R034 has no positive documented metric delta versus its parent; classification is FLAT-MIXED and submission gates passed=false.
