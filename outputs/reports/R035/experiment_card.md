# R035 experiment card

Derived factual companion to `decision_report.json`; feature importances are descriptive, not causal.

## Experiment

- Experiment ID: R035
- Parent: R029
- Incumbent: R017
- Timestamp: 2026-09-07T11:46:46.329453+00:00
- Immutable status: completed
- Hypothesis: Device?merchant familiarity captures whether a device is interacting with merchants that are normal for its own strictly prior history, adding a distinct pair-context signal beyond R029's customer, device, merchant, and merchant-composition features.
- Feature module: src.device_merchant
- Ordered added features: device_merchant_prior_count, device_merchant_share_of_device_history
- Feature count before/after: 52.000000 / 54.000000
- Inherited recipe: model=catboost; parameters={'loss_function': 'Logloss', 'iterations': 800, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'thread_count': 8, 'task_type': 'CPU', 'one_hot_max_size': 64}; parent config hash=35ce87bb3bc58be5cff23cea9418b7a954322563c1015f5e0727c1bb04fc3545
- Documented coherent change: append exactly the listed feature(s); no other generic-spec change is recorded.

## Metrics

| Metric | Score | Delta parent | Delta incumbent |
|---|---:|---:|---:|
| F1 | 0.796581 | -0.000990 | 0.006752 |
| F2 | 0.732443 | -0.000033 | 0.002848 |
| F2 early | 0.790953 | -0.000302 | 0.002008 |
| corrected F2 late | 0.679667 | 0.000141 | 0.003143 |
| July | 0.511224 | -0.000958 | 0.001943 |
| June 15-30 | 0.813800 | 0.000928 | 0.003667 |
| July 1-7 | 0.477210 | -0.000619 | 0.005005 |
| July 8-15 | 0.537156 | -0.001041 | 0.000100 |

## Decision

- Classification: FLAT-MIXED
- Submission gates passed: false
- Gate evidence: July_1_15=false, F2_late_corrected=true, F2=false, F1=true, July_1_7=true, July_8_15=true
- Early stopped: false
- Early-stop policy: {'F2': 0.01, 'F2_late_corrected': 0.005, 'July_1_15': 0.005}

## Feature importances

- Added features, F1: device_merchant_share_of_device_history (1.455290), device_merchant_prior_count (0.552456)
- Added features, F2: device_merchant_share_of_device_history (1.044030), device_merchant_prior_count (0.186886)
- Top 10 overall, F1: customer_location_count_share (11.328232), location (5.482712), customer_log_amount_minus_prior_mean (5.460143), customer_amount_to_prior_mean (5.179430), hour (4.865494), log_amount_bdt (4.575198), customer_device_count_share (4.536530), amount_bdt (3.575982), merchant_new_customer_share_24h (3.483729), customer_seconds_since_last (2.659481)
- Top 10 overall, F2: customer_location_count_share (10.222227), customer_log_amount_minus_prior_mean (10.192423), merchant_new_customer_share_24h (6.385707), location (4.856088), customer_device_count_share (4.234093), hour (3.590149), customer_device_prior_count (3.377077), customer_share_of_device_prior_transactions (3.110266), customer_amount_to_prior_mean (3.094666), device_prior_distinct_customer_count_excluding_current (3.076991)

## Certification and provenance

- Certification status: new
- Certification results: {'oracle': True, 'strict_past': True, 'equal_timestamp_isolation': True, 'permutation_invariance': True, 'duplicate_same_timestamp_pair': True, 'future_independence': True, 'label_independence': True, 'chunk_vs_whole': True, 'raw_frame_compatibility': True}
- Parent matrix parity / row-ID alignment: verified by pre-fit candidate checks
- Locked split signature: 6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47
- Train input fingerprint: {'sha256': '06823ea80cceade018b2428e0c7c9cb89e4efae45fcad8cbf61e8b27b350df20', 'rows': 731942, 'columns': ['transaction_id', 'customer_id', 'timestamp', 'amount_bdt', 'merchant_id', 'merchant_category', 'device_id', 'device_type', 'location', 'payment_method', 'transaction_type', 'account_age_days', 'fraud']}
- Test inference / submission: False / False
- Runtime seconds: 56.351435
- OOF: outputs/oof/R035.csv
- Model/config/spec/report/source: ['outputs/models/R035/fold_1.pkl', 'outputs/models/R035/fold_0.pkl']; outputs/reports/R035//config.json; specs\r035.json; outputs/reports/R035//decision_report.json; outputs/reports/R035//source
- Module hashes: {'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_history.py': '1e43ea1b15c7b86c4187917bdc35fe40f30ffbdd4695458d4f2125ae82b7c779', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_relationships.py': 'c83515d9c37621532aae8fc321b2c55bcf4271663f808a71af2b1090ad4ecf31', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\device_merchant.py': '21fc278c6d2edefa45f69610a63ef83f022d2f68b18fe4eeeea0fbc6af278a82'}
- Runner hashes: {'feature_experiment.py': '1c550f26933f32c9f585d12095b5c4f4f6b789c15a43495c64c937042b61b5c5', 'react_runner.py': '662ca45b1121307c5160726415d3fcd4a654109328c0d7261a445a5c284bc6e0'}
- Parent cache: {'used': True, 'path': 'outputs\\feature_matrices\\R029.pkl', 'matrix_sha256': 'ab767d22d77a44ef3514e8580d9287be720b210475037158514883c1c249771c', 'manifest_sha256': '433acf180bbe536d3d86d2a2cf559ae2ca4368548cb887ae56a601dbe02a028a'}

## Factual conclusion

R035 improved corrected F2 late, June 15-30 versus its parent; classification is FLAT-MIXED and submission gates passed=false.
