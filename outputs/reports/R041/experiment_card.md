# R041 experiment card

Derived factual companion to `decision_report.json`; feature importances are descriptive, not causal.

## Experiment

- Experiment ID: R041
- Parent: R029
- Incumbent: R017
- Timestamp: 2026-09-07T15:22:39.139259+00:00
- Immutable status: completed
- Hypothesis: Merchant-location historical support and merchant-normalized location share may capture spatial merchant behavior not represented by existing customer-location, merchant-global, or pair familiarity features.
- Feature module: src.merchant_location
- Ordered added features: merchant_location_prior_count, merchant_location_share_of_merchant_history
- Feature count before/after: 52.000000 / 54.000000
- Inherited recipe: model=catboost; parameters={'loss_function': 'Logloss', 'iterations': 800, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'thread_count': 8, 'task_type': 'CPU', 'one_hot_max_size': 64}; parent config hash=35ce87bb3bc58be5cff23cea9418b7a954322563c1015f5e0727c1bb04fc3545
- Documented coherent change: append exactly the listed feature(s); no other generic-spec change is recorded.

## Metrics

| Metric | Score | Delta parent | Delta incumbent |
|---|---:|---:|---:|
| F1 | 0.803311 | 0.005740 | 0.013482 |
| F2 | 0.734314 | 0.001838 | 0.004720 |
| F2 early | 0.794477 | 0.003222 | 0.005532 |
| corrected F2 late | 0.680395 | 0.000869 | 0.003871 |
| July | 0.512479 | 0.000297 | 0.003198 |
| June 15-30 | 0.814163 | 0.001291 | 0.004030 |
| July 1-7 | 0.477769 | -0.000059 | 0.005564 |
| July 8-15 | 0.539057 | 0.000860 | 0.002001 |

## Decision

- Classification: FLAT-MIXED
- Submission gates passed: true
- Gate evidence: July_1_15=true, F2_late_corrected=true, F2=true, F1=true, July_1_7=true, July_8_15=true
- Early stopped: false
- Early-stop policy: {'F2': 0.01, 'F2_late_corrected': 0.005, 'July_1_15': 0.005}

## Feature importances

- Added features, F1: merchant_location_share_of_merchant_history (5.250831), merchant_location_prior_count (1.326278)
- Added features, F2: merchant_location_share_of_merchant_history (4.228867), merchant_location_prior_count (1.230188)
- Top 10 overall, F1: customer_location_count_share (9.337036), customer_log_amount_minus_prior_mean (5.997583), merchant_location_share_of_merchant_history (5.250831), customer_amount_to_prior_mean (5.226987), customer_device_count_share (4.824172), hour (4.813673), amount_bdt (4.478505), log_amount_bdt (3.635329), merchant_new_customer_share_24h (3.495717), customer_location_prior_count (3.178447)
- Top 10 overall, F2: customer_log_amount_minus_prior_mean (9.983667), customer_location_count_share (9.277550), merchant_new_customer_share_24h (5.912244), merchant_location_share_of_merchant_history (4.228867), customer_device_count_share (4.184234), hour (3.622095), device_prior_distinct_customer_count_excluding_current (3.446913), customer_share_of_device_prior_transactions (3.133012), device_observed_age_seconds (3.061524), customer_amount_to_prior_mean (3.047276)

## Certification and provenance

- Certification status: new
- Certification results: {'oracle': True, 'strict_past': True, 'equal_timestamp_isolation': True, 'permutation_invariance': True, 'duplicate_same_timestamp_pair': True, 'future_independence': True, 'future_value_mutation_independence': True, 'label_independence': True, 'chunk_vs_whole': True, 'raw_frame_compatibility': True, 'requested_subset_order': True, 'zero_history_behavior': True, 'denominator_parity': True}
- Parent matrix parity / row-ID alignment: verified by pre-fit candidate checks
- Locked split signature: 6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47
- Train input fingerprint: {'sha256': '06823ea80cceade018b2428e0c7c9cb89e4efae45fcad8cbf61e8b27b350df20', 'rows': 731942, 'columns': ['transaction_id', 'customer_id', 'timestamp', 'amount_bdt', 'merchant_id', 'merchant_category', 'device_id', 'device_type', 'location', 'payment_method', 'transaction_type', 'account_age_days', 'fraud']}
- Test inference / submission: False / False
- Runtime seconds: 53.171222
- OOF: outputs/oof/R041.csv
- Model/config/spec/report/source: ['outputs/models/R041/fold_1.pkl', 'outputs/models/R041/fold_0.pkl']; outputs/reports/R041//config.json; specs\r041.json; outputs/reports/R041//decision_report.json; outputs/reports/R041//source
- Module hashes: {'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_history.py': '1e43ea1b15c7b86c4187917bdc35fe40f30ffbdd4695458d4f2125ae82b7c779', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_relationships.py': 'c83515d9c37621532aae8fc321b2c55bcf4271663f808a71af2b1090ad4ecf31', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\merchant_history.py': '08505e5b18f7722b71e5c9a5847478267a6cd5e6cb85311eb9a2f4c93999ae36', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\merchant_location.py': 'cb2e1079e74a1f839d7b1ef933b7319447084e52bd6084b32cd503479f9bcc9a'}
- Runner hashes: {'feature_experiment.py': '929442773a4c0a958be1bd89d0fcdfc0d9ab7d143cfc525dcf96c19dc01cda59', 'react_runner.py': '966fcd8e9355073f424ba7c8602f7aab713b8aa1f88eea4a917bbb36f04534b1'}
- Parent cache: {'used': True, 'path': 'outputs\\feature_matrices\\R029.pkl', 'matrix_sha256': 'ab767d22d77a44ef3514e8580d9287be720b210475037158514883c1c249771c', 'manifest_sha256': '433acf180bbe536d3d86d2a2cf559ae2ca4368548cb887ae56a601dbe02a028a'}
- Diagnostics: outputs\reports\R041\feature_diagnostics.json

## Factual conclusion

R041 improved F1, F2, F2 early, corrected F2 late, July, June 15-30, July 8-15 versus its parent; classification is FLAT-MIXED and submission gates passed=true.
