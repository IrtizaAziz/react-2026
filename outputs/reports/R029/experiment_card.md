# R029 experiment card

Derived factual companion to `decision_report.json`; feature importances are descriptive, not causal.

## Experiment

- Experiment ID: R029
- Parent: R026
- Incumbent: R017
- Timestamp: 2026-09-07T10:33:51.778387+00:00
- Immutable status: completed
- Hypothesis: The time since a merchant last acquired a new customer captures temporal customer-composition change beyond R026's 24-hour new-customer share, while the broader R025 package may have diluted this recency signal.
- Feature module: src.merchant_new_customers
- Ordered added features: merchant_seconds_since_last_new_customer
- Feature count before/after: 51.000000 / 52.000000
- Inherited recipe: model=catboost; parameters={'loss_function': 'Logloss', 'iterations': 800, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'thread_count': 8, 'task_type': 'CPU', 'one_hot_max_size': 64}; parent config hash=cc92c23eea4ed65e1dc962f266002bf8ac5ea8ba90f1037165740e8f7d4cddc3
- Documented coherent change: append exactly the listed feature(s); no other generic-spec change is recorded.

## Metrics

| Metric | Score | Delta parent | Delta incumbent |
|---|---:|---:|---:|
| F1 | 0.797571 | 0.002890 | 0.007742 |
| F2 | 0.732476 | 0.000686 | 0.002881 |
| F2 early | 0.791255 | 0.000689 | 0.002310 |
| corrected F2 late | 0.679526 | 0.000643 | 0.003002 |
| July | 0.512182 | 0.001507 | 0.002901 |
| June 15-30 | 0.812873 | 0.000251 | 0.002739 |
| July 1-7 | 0.477828 | 0.003746 | 0.005623 |
| July 8-15 | 0.538197 | -0.000027 | 0.001140 |

## Decision

- Classification: FLAT-MIXED
- Submission gates passed: false
- Gate evidence: July_1_15=false, F2_late_corrected=true, F2=false, F1=true, July_1_7=true, July_8_15=true
- Early stopped: false
- Early-stop policy: {'F2': 0.01, 'F2_late_corrected': 0.005, 'July_1_15': 0.005}

## Feature importances

- Added features, F1: merchant_seconds_since_last_new_customer (2.365375)
- Added features, F2: merchant_seconds_since_last_new_customer (1.860608)
- Top 10 overall, F1: customer_location_count_share (11.374670), customer_log_amount_minus_prior_mean (7.818795), location (5.603087), hour (5.162579), customer_device_count_share (4.614896), log_amount_bdt (4.377713), amount_bdt (4.057328), merchant_new_customer_share_24h (3.467494), customer_amount_to_prior_mean (2.959361), customer_seconds_since_last (2.661582)
- Top 10 overall, F2: customer_log_amount_minus_prior_mean (10.236740), customer_location_count_share (9.609729), merchant_new_customer_share_24h (6.243187), customer_device_count_share (5.815054), location (4.941841), hour (3.166410), log_amount_bdt (3.067280), customer_share_of_device_prior_transactions (3.019978), amount_bdt (2.986035), customer_amount_to_prior_mean (2.978542)

## Certification and provenance

- Certification status: reused
- Certification results: {'strict_past': True, 'oracle': True, 'equal_timestamp_isolation': True, 'permutation_invariance': True, 'future_independence': True, 'label_independence': True, 'chunk_vs_whole': True}
- Parent matrix parity / row-ID alignment: verified by pre-fit candidate checks
- Locked split signature: 6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47
- Train input fingerprint: {'sha256': '06823ea80cceade018b2428e0c7c9cb89e4efae45fcad8cbf61e8b27b350df20', 'rows': 731942, 'columns': ['transaction_id', 'customer_id', 'timestamp', 'amount_bdt', 'merchant_id', 'merchant_category', 'device_id', 'device_type', 'location', 'payment_method', 'transaction_type', 'account_age_days', 'fraud']}
- Test inference / submission: False / False
- Runtime seconds: 45.327452
- OOF: outputs/oof/R029.csv
- Model/config/spec/report/source: ['outputs/models/R029/fold_1.pkl', 'outputs/models/R029/fold_0.pkl']; outputs/reports/R029//config.json; specs\r029.json; outputs/reports/R029//decision_report.json; outputs/reports/R029//source
- Module hashes: {'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\merchant_new_customers.py': '0932f5473e640259972d5628a989c77becbcd13ce25fc0fd3740e123bafef707'}
- Runner hashes: {'feature_experiment.py': 'd90a72384d57399717785cdcabc5a985b7ced60919fcd5d257d30aa0ef626d2d', 'react_runner.py': 'b24854c977f37d9e2b3466639f5d07e6e968184bffc13b3577b6d9cc4ca05a5f'}
- Parent cache: {'used': True, 'path': 'outputs\\feature_matrices\\R026.pkl', 'matrix_sha256': '54881bb19b4d9c306fc3731247a8b9ba3dc10170541d2e86b3f8018dbf454887', 'manifest_sha256': '362290205af1e8bc3dd5068fff11c029c68f5b8f017ed95a8554357f06397282'}

## Reference-run analysis

- R025: compatible=True; deltas={'F1': -0.0004018763874449327, 'F2': 0.001545507463098339, 'F2_early': 0.0012539331430565515, 'F2_late_corrected': 0.002035307121597718, 'July_1_15': 0.002324964613475089}

## Factual conclusion

R029 improved F1, F2, F2 early, corrected F2 late, July, June 15-30, July 1-7 versus its parent; classification is FLAT-MIXED and submission gates passed=false.
