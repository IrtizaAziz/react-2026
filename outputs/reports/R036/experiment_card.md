# R036 experiment card

Derived factual companion to `decision_report.json`; feature importances are descriptive, not causal.

## Experiment

- Experiment ID: R036
- Parent: R029
- Incumbent: R017
- Timestamp: 2026-09-07T11:50:49.971048+00:00
- Immutable status: completed
- Hypothesis: The share of a device's strictly prior history associated with the current merchant captures device-specific merchant familiarity beyond R029, while the absolute device-merchant count in R035 may have added redundant scale information.
- Feature module: src.device_merchant
- Ordered added features: device_merchant_share_of_device_history
- Feature count before/after: 52.000000 / 53.000000
- Inherited recipe: model=catboost; parameters={'loss_function': 'Logloss', 'iterations': 800, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'thread_count': 8, 'task_type': 'CPU', 'one_hot_max_size': 64}; parent config hash=35ce87bb3bc58be5cff23cea9418b7a954322563c1015f5e0727c1bb04fc3545
- Documented coherent change: append exactly the listed feature(s); no other generic-spec change is recorded.

## Metrics

| Metric | Score | Delta parent | Delta incumbent |
|---|---:|---:|---:|
| F1 | 0.795057 | -0.002514 | 0.005228 |
| F2 | 0.731711 | -0.000765 | 0.002117 |
| F2 early | 0.790402 | -0.000853 | 0.001457 |
| corrected F2 late | 0.678841 | -0.000685 | 0.002317 |
| July | 0.511262 | -0.000920 | 0.001981 |
| June 15-30 | 0.812140 | -0.000733 | 0.002007 |
| July 1-7 | 0.475452 | -0.002376 | 0.003248 |
| July 8-15 | 0.538652 | 0.000456 | 0.001596 |

## Decision

- Classification: LOSE
- Submission gates passed: false
- Gate evidence: July_1_15=false, F2_late_corrected=true, F2=false, F1=true, July_1_7=true, July_8_15=true
- Early stopped: false
- Early-stop policy: {'F2': 0.01, 'F2_late_corrected': 0.005, 'July_1_15': 0.005}

## Feature importances

- Added features, F1: device_merchant_share_of_device_history (1.190782)
- Added features, F2: device_merchant_share_of_device_history (1.059433)
- Top 10 overall, F1: customer_location_count_share (10.410708), customer_log_amount_minus_prior_mean (6.111441), hour (5.312211), location (5.088557), log_amount_bdt (5.068544), customer_device_count_share (4.392898), customer_location_prior_count (3.741328), amount_bdt (3.428619), merchant_new_customer_share_24h (3.330369), customer_amount_to_prior_mean (3.243225)
- Top 10 overall, F2: customer_location_count_share (11.538424), customer_log_amount_minus_prior_mean (10.521551), merchant_new_customer_share_24h (5.915397), location (5.336683), customer_device_count_share (5.217656), log_amount_bdt (3.538489), customer_share_of_device_prior_transactions (3.060230), device_prior_distinct_customer_count_excluding_current (3.031302), customer_amount_to_prior_mean (2.925478), hour (2.909720)

## Certification and provenance

- Certification status: reused
- Certification results: {'oracle': True, 'strict_past': True, 'equal_timestamp_isolation': True, 'permutation_invariance': True, 'duplicate_same_timestamp_pair': True, 'future_independence': True, 'label_independence': True, 'chunk_vs_whole': True, 'raw_frame_compatibility': True}
- Parent matrix parity / row-ID alignment: verified by pre-fit candidate checks
- Locked split signature: 6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47
- Train input fingerprint: {'sha256': '06823ea80cceade018b2428e0c7c9cb89e4efae45fcad8cbf61e8b27b350df20', 'rows': 731942, 'columns': ['transaction_id', 'customer_id', 'timestamp', 'amount_bdt', 'merchant_id', 'merchant_category', 'device_id', 'device_type', 'location', 'payment_method', 'transaction_type', 'account_age_days', 'fraud']}
- Test inference / submission: False / False
- Runtime seconds: 54.773032
- OOF: outputs/oof/R036.csv
- Model/config/spec/report/source: ['outputs/models/R036/fold_1.pkl', 'outputs/models/R036/fold_0.pkl']; outputs/reports/R036//config.json; specs\r036.json; outputs/reports/R036//decision_report.json; outputs/reports/R036//source
- Module hashes: {'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_history.py': '1e43ea1b15c7b86c4187917bdc35fe40f30ffbdd4695458d4f2125ae82b7c779', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_relationships.py': 'c83515d9c37621532aae8fc321b2c55bcf4271663f808a71af2b1090ad4ecf31', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\device_merchant.py': '21fc278c6d2edefa45f69610a63ef83f022d2f68b18fe4eeeea0fbc6af278a82'}
- Runner hashes: {'feature_experiment.py': '1c550f26933f32c9f585d12095b5c4f4f6b789c15a43495c64c937042b61b5c5', 'react_runner.py': '662ca45b1121307c5160726415d3fcd4a654109328c0d7261a445a5c284bc6e0'}
- Parent cache: {'used': True, 'path': 'outputs\\feature_matrices\\R029.pkl', 'matrix_sha256': 'ab767d22d77a44ef3514e8580d9287be720b210475037158514883c1c249771c', 'manifest_sha256': '433acf180bbe536d3d86d2a2cf559ae2ca4368548cb887ae56a601dbe02a028a'}

## Reference-run analysis

- R035: compatible=True; deltas={'F2': -0.0007317141329857835, 'F1': -0.0015238951985986349, 'F2_early': -0.0005511021154244178, 'F2_late_corrected': -0.0008261918846383942}

## Factual conclusion

R036 improved July 8-15 versus its parent; classification is LOSE and submission gates passed=false.
