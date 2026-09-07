# R032 experiment card

Derived factual companion to `decision_report.json`; feature importances are descriptive, not causal.

## Experiment

- Experiment ID: R032
- Parent: R026
- Incumbent: R017
- Timestamp: 2026-09-07T11:00:55.403872+00:00
- Immutable status: completed
- Hypothesis: The number of other customers using a device in the prior 24 hours captures acute cross-customer device sharing beyond R026, while the broader R015 recent-sharing package may have diluted this concentrated signal.
- Feature module: src.device_recent_sharing
- Ordered added features: device_other_customers_24h
- Feature count before/after: 51.000000 / 52.000000
- Inherited recipe: model=catboost; parameters={'loss_function': 'Logloss', 'iterations': 800, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'thread_count': 8, 'task_type': 'CPU', 'one_hot_max_size': 64}; parent config hash=cc92c23eea4ed65e1dc962f266002bf8ac5ea8ba90f1037165740e8f7d4cddc3
- Documented coherent change: append exactly the listed feature(s); no other generic-spec change is recorded.

## Metrics

| Metric | Score | Delta parent | Delta incumbent |
|---|---:|---:|---:|
| F1 | 0.795375 | 0.000693 | 0.005546 |
| F2 | 0.732881 | 0.001091 | 0.003287 |
| F2 early | 0.792169 | 0.001603 | 0.003224 |
| corrected F2 late | 0.679135 | 0.000252 | 0.002610 |
| July | 0.511996 | 0.001321 | 0.002715 |
| June 15-30 | 0.812535 | -0.000087 | 0.002401 |
| July 1-7 | 0.476164 | 0.002082 | 0.003959 |
| July 8-15 | 0.539082 | 0.000858 | 0.002026 |

## Decision

- Classification: FLAT-MIXED
- Submission gates passed: false
- Gate evidence: July_1_15=false, F2_late_corrected=true, F2=true, F1=true, July_1_7=true, July_8_15=true
- Early stopped: false
- Early-stop policy: {'F2': 0.01, 'F2_late_corrected': 0.005, 'July_1_15': 0.005}

## Feature importances

- Added features, F1: device_other_customers_24h (0.547955)
- Added features, F2: device_other_customers_24h (0.989446)
- Top 10 overall, F1: customer_location_count_share (11.025530), customer_log_amount_minus_prior_mean (7.795154), hour (5.755577), location (5.345744), customer_device_count_share (4.917897), amount_bdt (4.635498), log_amount_bdt (3.841660), merchant_new_customer_share_24h (3.274454), customer_amount_to_prior_mean (3.266209), customer_seconds_since_last (2.875075)
- Top 10 overall, F2: customer_log_amount_minus_prior_mean (10.532608), customer_location_count_share (9.535659), merchant_new_customer_share_24h (5.838019), location (5.048270), customer_device_count_share (3.941623), customer_device_prior_count (3.728873), customer_share_of_device_prior_transactions (3.595869), customer_amount_to_prior_mean (3.461713), hour (3.444958), log_amount_bdt (2.965920)

## Certification and provenance

- Certification status: new
- Certification results: {'strict_past': True, 'equal_timestamp_isolation': True, 'oracle': True, 'permutation_invariance': True, 'duplicate_same_timestamp_pair': True, 'future_independence': True, 'label_independence': True, 'chunk_vs_whole': True, 'excluding_current_customer': True, 'raw_frame_compatibility': True}
- Parent matrix parity / row-ID alignment: verified by pre-fit candidate checks
- Locked split signature: 6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47
- Train input fingerprint: {'sha256': '06823ea80cceade018b2428e0c7c9cb89e4efae45fcad8cbf61e8b27b350df20', 'rows': 731942, 'columns': ['transaction_id', 'customer_id', 'timestamp', 'amount_bdt', 'merchant_id', 'merchant_category', 'device_id', 'device_type', 'location', 'payment_method', 'transaction_type', 'account_age_days', 'fraud']}
- Test inference / submission: False / False
- Runtime seconds: 57.508442
- OOF: outputs/oof/R032.csv
- Model/config/spec/report/source: ['outputs/models/R032/fold_1.pkl', 'outputs/models/R032/fold_0.pkl']; outputs/reports/R032//config.json; specs\r032.json; outputs/reports/R032//decision_report.json; outputs/reports/R032//source
- Module hashes: {'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_history.py': '1e43ea1b15c7b86c4187917bdc35fe40f30ffbdd4695458d4f2125ae82b7c779', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_relationships.py': 'c83515d9c37621532aae8fc321b2c55bcf4271663f808a71af2b1090ad4ecf31', 'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\device_recent_sharing.py': '00933ba3078c84aba8db39a45b728a5ecffaa81aecda78673a7dfcfa458f2836'}
- Runner hashes: {'feature_experiment.py': '1c550f26933f32c9f585d12095b5c4f4f6b789c15a43495c64c937042b61b5c5', 'react_runner.py': '662ca45b1121307c5160726415d3fcd4a654109328c0d7261a445a5c284bc6e0'}
- Parent cache: {'used': True, 'path': 'outputs\\feature_matrices\\R026.pkl', 'matrix_sha256': '54881bb19b4d9c306fc3731247a8b9ba3dc10170541d2e86b3f8018dbf454887', 'manifest_sha256': '362290205af1e8bc3dd5068fff11c029c68f5b8f017ed95a8554357f06397282'}

## Reference-run analysis

- R015: compatible=False; deltas=UNKNOWN

## Factual conclusion

R032 improved F1, F2, F2 early, corrected F2 late, July, July 1-7, July 8-15 versus its parent; classification is FLAT-MIXED and submission gates passed=false.
