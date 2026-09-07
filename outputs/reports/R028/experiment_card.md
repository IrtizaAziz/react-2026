# R028 experiment card

Derived factual companion to `decision_report.json`; feature importances are descriptive, not causal.

## Experiment

- Experiment ID: R028
- Parent: R026
- Incumbent: R017
- Timestamp: 2026-09-07T10:27:35.722914+00:00
- Immutable status: completed
- Hypothesis: The time since a customer last formed a new merchant relationship captures relationship-change recency beyond R026, while the rolling new-merchant count and share features bundled in R023 may have diluted this signal.
- Feature module: src.customer_new_merchants
- Ordered added features: customer_seconds_since_last_new_merchant
- Feature count before/after: 51.000000 / 52.000000
- Inherited recipe: model=catboost; parameters={'loss_function': 'Logloss', 'iterations': 800, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'thread_count': 8, 'task_type': 'CPU', 'one_hot_max_size': 64}; parent config hash=cc92c23eea4ed65e1dc962f266002bf8ac5ea8ba90f1037165740e8f7d4cddc3
- Documented coherent change: append exactly the listed feature(s); no other generic-spec change is recorded.

## Metrics

| Metric | Score | Delta parent | Delta incumbent |
|---|---:|---:|---:|
| F1 | 0.796804 | 0.002122 | 0.006975 |
| F2 | 0.731824 | 0.000034 | 0.002230 |
| F2 early | 0.790649 | 0.000083 | 0.001704 |
| corrected F2 late | 0.678833 | -0.000050 | 0.002309 |
| July | 0.512080 | 0.001405 | 0.002799 |
| June 15-30 | 0.811614 | -0.001008 | 0.001481 |
| July 1-7 | 0.476324 | 0.002242 | 0.004119 |
| July 8-15 | 0.539024 | 0.000800 | 0.001967 |

## Decision

- Classification: FLAT-MIXED
- Submission gates passed: false
- Gate evidence: July_1_15=false, F2_late_corrected=true, F2=false, F1=true, July_1_7=true, July_8_15=true
- Early stopped: false
- Early-stop policy: {'F2': 0.01, 'F2_late_corrected': 0.005, 'July_1_15': 0.005}

## Feature importances

- Added features, F1: customer_seconds_since_last_new_merchant (1.886157)
- Added features, F2: customer_seconds_since_last_new_merchant (1.918307)
- Top 10 overall, F1: customer_location_count_share (10.716499), customer_log_amount_minus_prior_mean (7.813839), hour (5.834468), location (5.236906), customer_device_count_share (4.698660), log_amount_bdt (4.424863), amount_bdt (4.310979), merchant_new_customer_share_24h (3.672283), customer_location_new (3.191636), merchant_seconds_since_last (2.569140)
- Top 10 overall, F2: customer_log_amount_minus_prior_mean (10.106073), customer_location_count_share (9.875764), merchant_new_customer_share_24h (5.763744), customer_device_count_share (4.941337), location (4.834103), hour (3.491805), customer_device_prior_count (3.410342), customer_share_of_device_prior_transactions (3.372079), amount_bdt (2.951398), device_prior_distinct_customer_count_excluding_current (2.908168)

## Certification and provenance

- Certification status: new
- Certification results: {'strict_past': True, 'oracle': True, 'equal_timestamp_isolation': True, 'permutation_invariance': True, 'duplicate_same_timestamp_customer_merchant': True, 'future_independence': True, 'label_independence': True, 'chunk_vs_whole': True}
- Parent matrix parity / row-ID alignment: verified by pre-fit candidate checks
- Locked split signature: 6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47
- Train input fingerprint: {'sha256': '06823ea80cceade018b2428e0c7c9cb89e4efae45fcad8cbf61e8b27b350df20', 'rows': 731942, 'columns': ['transaction_id', 'customer_id', 'timestamp', 'amount_bdt', 'merchant_id', 'merchant_category', 'device_id', 'device_type', 'location', 'payment_method', 'transaction_type', 'account_age_days', 'fraud']}
- Test inference / submission: False / False
- Runtime seconds: 57.499506
- OOF: outputs/oof/R028.csv
- Model/config/spec/report/source: ['outputs/models/R028/fold_1.pkl', 'outputs/models/R028/fold_0.pkl']; outputs/reports/R028//config.json; specs\r028.json; outputs/reports/R028//decision_report.json; outputs/reports/R028//source
- Module hashes: {'C:\\Users\\IRTIZA\\Downloads\\Datathon\\react-2026\\src\\customer_new_merchants.py': '828a7ee1d9b247c6c58ab959ba6561a717b06d3ddce4baf1d29854412d39d422'}
- Runner hashes: {'feature_experiment.py': 'd90a72384d57399717785cdcabc5a985b7ced60919fcd5d257d30aa0ef626d2d', 'react_runner.py': 'b24854c977f37d9e2b3466639f5d07e6e968184bffc13b3577b6d9cc4ca05a5f'}
- Parent cache: {'used': True, 'path': 'outputs\\feature_matrices\\R026.pkl', 'matrix_sha256': '54881bb19b4d9c306fc3731247a8b9ba3dc10170541d2e86b3f8018dbf454887', 'manifest_sha256': '362290205af1e8bc3dd5068fff11c029c68f5b8f017ed95a8554357f06397282'}

## Reference-run analysis

- R023: compatible=True; deltas={'F1': 0.0065943859964373575, 'F2': 0.003089628447968895, 'F2_early': 0.002997763203982595, 'F2_late_corrected': 0.002809901547422222, 'July_1_15': 0.0035165517980484395}

## Factual conclusion

R028 improved F1, F2, F2 early, July, July 1-7, July 8-15 versus its parent; classification is FLAT-MIXED and submission gates passed=false.
