# R043 experiment card

Derived factual companion to `decision_report.json`; feature importances are descriptive, not causal.

## Experiment

- Experiment ID: R043
- Parent: R029
- Incumbent: R029
- Timestamp: 2026-09-07T16:14:01.589280+00:00
- Immutable status: completed
- Hypothesis: A transaction whose calendar-day implied account creation date sharply disagrees with its customer's strictly-prior modal implied creation day may expose account-identity metadata inconsistency associated with fraud.
- Feature module: src.customer_account_consistency
- Ordered added features: customer_creation_day_signed_deviation, customer_creation_day_abs_deviation, customer_creation_day_mode_share
- Feature count before/after: 52.000000 / 55.000000
- Inherited recipe: model=catboost; parameters={'loss_function': 'Logloss', 'iterations': 800, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'thread_count': 8, 'task_type': 'CPU', 'one_hot_max_size': 64}; parent config hash=35ce87bb3bc58be5cff23cea9418b7a954322563c1015f5e0727c1bb04fc3545
- Documented coherent change: append exactly the listed feature(s); no other generic-spec change is recorded.

## Metrics

| Metric | Score | Delta parent | Delta incumbent |
|---|---:|---:|---:|
| F1 | 0.807398 | 0.009827 | 0.009827 |
| F2 | 0.733180 | 0.000705 | 0.000705 |
| F2 early | 0.793168 | 0.001913 | 0.001913 |
| corrected F2 late | 0.678818 | -0.000708 | -0.000708 |
| July | 0.511749 | -0.000432 | -0.000432 |
| June 15-30 | 0.812331 | -0.000542 | -0.000542 |
| July 1-7 | 0.482066 | 0.004238 | 0.004238 |
| July 8-15 | 0.534496 | -0.003700 | -0.003700 |

## Decision

- Classification: LOSE
- Submission gates passed: false
- Gate evidence: July_1_15=false, F2_late_corrected=false, F2=false, F1=true, July_1_7=true, July_8_15=false
- Early stopped: false
- Early-stop policy: {'F2': 0.01, 'F2_late_corrected': 0.005, 'July_1_15': 0.005}

## Feature importances

- Added features, F1: customer_creation_day_abs_deviation (1.203303), customer_creation_day_mode_share (0.411971), customer_creation_day_signed_deviation (0.155796)
- Added features, F2: customer_creation_day_abs_deviation (2.265076), customer_creation_day_mode_share (0.926925), customer_creation_day_signed_deviation (0.238022)
- Top 10 overall, F1: customer_location_count_share (11.866907), customer_log_amount_minus_prior_mean (5.675497), location (5.380542), log_amount_bdt (5.153423), customer_amount_to_prior_mean (4.961419), amount_bdt (4.679574), hour (4.678787), customer_device_prior_count (4.278188), customer_device_count_share (3.897696), customer_share_of_device_prior_transactions (2.771426)
- Top 10 overall, F2: customer_log_amount_minus_prior_mean (11.017003), customer_location_count_share (10.361389), customer_device_count_share (5.533416), location (5.491609), customer_share_of_device_prior_transactions (4.039331), customer_device_prior_count (3.771970), amount_bdt (3.505457), customer_location_new (3.179193), hour (3.164985), device_prior_distinct_customer_count_excluding_current (3.044080)

## Certification and provenance

- Certification status: new
- Certification results: {'oracle': True, 'strict_past': True, 'equal_timestamp_isolation': True, 'permutation_invariance': True, 'duplicate_same_timestamp_pair': True, 'future_independence': True, 'future_value_mutation_independence': True, 'label_independence': True, 'raw_frame_compatibility': True, 'requested_subset_order': True, 'zero_history_behavior': True, 'modal_tie_earliest': True, 'mode_robustness': True, 'calendar_day_semantics': True, 'denominator_parity': True, 'chunk_vs_whole': True}
- Parent matrix parity / row-ID alignment: verified by pre-fit candidate checks
- Locked split signature: 6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47
- Train input fingerprint: {'sha256': '06823ea80cceade018b2428e0c7c9cb89e4efae45fcad8cbf61e8b27b350df20', 'rows': 731942, 'columns': ['transaction_id', 'customer_id', 'timestamp', 'amount_bdt', 'merchant_id', 'merchant_category', 'device_id', 'device_type', 'location', 'payment_method', 'transaction_type', 'account_age_days', 'fraud']}
- Test inference / submission: False / False
- Runtime seconds: 59.831144
- OOF: outputs/oof/R043.csv
- Model/config/spec/report/source: ['outputs/models/R043/fold_1.pkl', 'outputs/models/R043/fold_0.pkl']; outputs/reports/R043//config.json; specs\r043.json; outputs/reports/R043//decision_report.json; outputs/reports/R043//source
- Module hashes: {'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_account_consistency.py': 'affb1d68628027971c8308cf71f95b6e3392cb0181f6b8dee2efdee7649a1ea5', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_history.py': '1e43ea1b15c7b86c4187917bdc35fe40f30ffbdd4695458d4f2125ae82b7c779'}
- Runner hashes: {'feature_experiment.py': '929442773a4c0a958be1bd89d0fcdfc0d9ab7d143cfc525dcf96c19dc01cda59', 'react_runner.py': '966fcd8e9355073f424ba7c8602f7aab713b8aa1f88eea4a917bbb36f04534b1'}
- Parent cache: {'used': True, 'path': 'outputs\\feature_matrices\\R029.pkl', 'matrix_sha256': 'ab767d22d77a44ef3514e8580d9287be720b210475037158514883c1c249771c', 'manifest_sha256': '433acf180bbe536d3d86d2a2cf559ae2ca4368548cb887ae56a601dbe02a028a'}
- Diagnostics: outputs\reports\R043\feature_diagnostics.json

## Reference-run analysis

- R040: compatible=False; deltas=UNKNOWN

## Factual conclusion

R043 improved F1, F2, F2 early, July 1-7 versus its parent; classification is LOSE and submission gates passed=false.
