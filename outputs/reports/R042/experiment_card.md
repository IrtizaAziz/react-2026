# R042 experiment card

Derived factual companion to `decision_report.json`; feature importances are descriptive, not causal.

## Experiment

- Experiment ID: R042
- Parent: R029
- Incumbent: R029
- Timestamp: 2026-09-07T15:58:30.904152+00:00
- Immutable status: completed
- Hypothesis: A merchant may become suspicious when its transaction activity rises sharply relative to its immediately preceding short-term baseline; explicit last-hour volume and adjacent-hour log-rate change may capture temporal dynamics beyond R029.
- Feature module: src.merchant_acceleration
- Ordered added features: merchant_transactions_1h, merchant_log_rate_change_1h_vs_previous_1h
- Feature count before/after: 52.000000 / 54.000000
- Inherited recipe: model=catboost; parameters={'loss_function': 'Logloss', 'iterations': 800, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'thread_count': 8, 'task_type': 'CPU', 'one_hot_max_size': 64}; parent config hash=35ce87bb3bc58be5cff23cea9418b7a954322563c1015f5e0727c1bb04fc3545
- Documented coherent change: append exactly the listed feature(s); no other generic-spec change is recorded.

## Metrics

| Metric | Score | Delta parent | Delta incumbent |
|---|---:|---:|---:|
| F1 | 0.797093 | -0.000478 | -0.000478 |
| F2 | 0.732356 | -0.000119 | -0.000119 |
| F2 early | 0.791463 | 0.000208 | 0.000208 |
| corrected F2 late | 0.678787 | -0.000739 | -0.000739 |
| July | 0.510591 | -0.001591 | -0.001591 |
| June 15-30 | 0.812716 | -0.000157 | -0.000157 |
| July 1-7 | 0.476445 | -0.001383 | -0.001383 |
| July 8-15 | 0.536712 | -0.001485 | -0.001485 |

## Decision

- Classification: FLAT-MIXED
- Submission gates passed: false
- Gate evidence: July_1_15=false, F2_late_corrected=false, F2=false, F1=true, July_1_7=true, July_8_15=true
- Early stopped: false
- Early-stop policy: {'F2': 0.01, 'F2_late_corrected': 0.005, 'July_1_15': 0.005}

## Feature importances

- Added features, F1: merchant_transactions_1h (1.858514), merchant_log_rate_change_1h_vs_previous_1h (1.040734)
- Added features, F2: merchant_transactions_1h (2.284216), merchant_log_rate_change_1h_vs_previous_1h (0.511285)
- Top 10 overall, F1: customer_location_count_share (10.680529), customer_log_amount_minus_prior_mean (6.363263), location (5.440536), customer_device_count_share (5.133151), customer_amount_to_prior_mean (4.970434), hour (4.822751), amount_bdt (4.148092), log_amount_bdt (3.713135), customer_location_new (3.311863), merchant_new_customer_share_24h (3.082015)
- Top 10 overall, F2: customer_location_count_share (10.579885), customer_log_amount_minus_prior_mean (9.594639), merchant_new_customer_share_24h (5.643106), location (4.865823), customer_device_count_share (4.126296), hour (3.549191), device_prior_distinct_customer_count_excluding_current (3.443223), customer_share_of_device_prior_transactions (3.261797), customer_amount_to_prior_mean (3.159077), amount_bdt (3.074693)

## Certification and provenance

- Certification status: new
- Certification results: {'oracle': True, 'strict_past': True, 'equal_timestamp_isolation': True, 'permutation_invariance': True, 'duplicate_same_timestamp_pair': True, 'future_independence': True, 'future_value_mutation_independence': True, 'label_independence': True, 'chunk_vs_whole': True, 'raw_frame_compatibility': True, 'requested_subset_order': True, 'zero_history': True, 'exact_window_boundaries': True}
- Parent matrix parity / row-ID alignment: verified by pre-fit candidate checks
- Locked split signature: 6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47
- Train input fingerprint: {'sha256': '06823ea80cceade018b2428e0c7c9cb89e4efae45fcad8cbf61e8b27b350df20', 'rows': 731942, 'columns': ['transaction_id', 'customer_id', 'timestamp', 'amount_bdt', 'merchant_id', 'merchant_category', 'device_id', 'device_type', 'location', 'payment_method', 'transaction_type', 'account_age_days', 'fraud']}
- Test inference / submission: False / False
- Runtime seconds: 58.675365
- OOF: outputs\oof\R042.csv
- Model/config/spec/report/source: ['outputs/models/R042/fold_1.pkl', 'outputs/models/R042/fold_0.pkl']; outputs\reports\R042/config.json; C:\Users\IRTIZA\Downloads\Datathon\react-2026\specs\r042.json; outputs\reports\R042/decision_report.json; outputs\reports\R042/source
- Module hashes: {'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_history.py': '1e43ea1b15c7b86c4187917bdc35fe40f30ffbdd4695458d4f2125ae82b7c779', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\merchant_acceleration.py': '5db3ee72cdbb44635a1be78b56be39226abebf5193050eb1b8e1458a8b15d6c6'}
- Runner hashes: {'feature_experiment.py': '929442773a4c0a958be1bd89d0fcdfc0d9ab7d143cfc525dcf96c19dc01cda59', 'react_runner.py': '966fcd8e9355073f424ba7c8602f7aab713b8aa1f88eea4a917bbb36f04534b1'}
- Parent cache: {'used': True, 'path': 'outputs\\feature_matrices\\R029.pkl', 'matrix_sha256': 'ab767d22d77a44ef3514e8580d9287be720b210475037158514883c1c249771c', 'manifest_sha256': '433acf180bbe536d3d86d2a2cf559ae2ca4368548cb887ae56a601dbe02a028a'}
- Diagnostics: outputs\reports\R042\feature_diagnostics.json

## Reference-run analysis

- R040: compatible=False; deltas=UNKNOWN

## Factual conclusion

R042 improved F2 early versus its parent; classification is FLAT-MIXED and submission gates passed=false.
