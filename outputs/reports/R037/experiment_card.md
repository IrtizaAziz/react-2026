# R037 experiment card

Derived factual companion to `decision_report.json`; feature importances are descriptive, not causal.

## Experiment

- Experiment ID: R037
- Parent: R029
- Incumbent: R017
- Timestamp: 2026-09-07T13:21:39.758680+00:00
- Immutable status: completed
- Hypothesis: LightGBM on R029's exact 52-feature causal representation may improve temporal fraud ranking or provide complementary OOF rankings versus CatBoost without changing the information set.
- Feature module: 
- Ordered added features: UNKNOWN
- Feature count before/after: 52.000000 / 52.000000
- Inherited recipe: model=lightgbm; parameters={'objective': 'binary', 'n_estimators': 800, 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 100, 'reg_lambda': 5, 'colsample_bytree': 0.9, 'subsample': 0.8, 'subsample_freq': 1, 'random_state': 42, 'n_jobs': 8, 'deterministic': True, 'force_col_wise': True, 'verbosity': -1}; parent config hash=35ce87bb3bc58be5cff23cea9418b7a954322563c1015f5e0727c1bb04fc3545
- Documented coherent change: append exactly the listed feature(s); no other generic-spec change is recorded.

## Metrics

| Metric | Score | Delta parent | Delta incumbent |
|---|---:|---:|---:|
| F1 | 0.797246 | -0.000325 | 0.007417 |
| F2 | 0.732080 | -0.000396 | 0.002485 |
| F2 early | 0.789622 | -0.001633 | 0.000677 |
| corrected F2 late | 0.679810 | 0.000283 | 0.003285 |
| July | 0.513743 | 0.001561 | 0.004462 |
| June 15-30 | 0.812549 | -0.000323 | 0.002416 |
| July 1-7 | 0.485940 | 0.008112 | 0.013735 |
| July 8-15 | 0.535149 | -0.003047 | -0.001907 |

## Decision

- Classification: COMPETITIVE-BLEND-ELIGIBLE
- Submission gates passed: false
- Gate evidence: July_1_15=true, F2_late_corrected=true, F2=false, F1=true, July_1_7=true, July_8_15=true
- Early stopped: false
- Early-stop policy: {'F2': 0.01, 'F2_late_corrected': 0.005, 'July_1_15': 0.005}

## Feature importances

- Added features, F1: UNKNOWN
- Added features, F2: UNKNOWN
- Top 10 overall, F1: amount_bdt (1180.000000), account_age_days (1170.000000), device_seconds_since_last (1008.000000), merchant_new_customer_share_24h (911.000000), merchant_seconds_since_last_new_customer (889.000000), device_observed_age_seconds (864.000000), merchant_seconds_since_last (852.000000), hour (833.000000), customer_prior_std_log_amount (784.000000), customer_seconds_since_last (762.000000)
- Top 10 overall, F2: amount_bdt (963.000000), account_age_days (947.000000), device_seconds_since_last (877.000000), merchant_seconds_since_last (863.000000), device_observed_age_seconds (825.000000), customer_prior_std_log_amount (822.000000), merchant_new_customer_share_24h (797.000000), customer_seconds_since_last (794.000000), merchant_seconds_since_last_new_customer (780.000000), customer_log_amount_minus_prior_mean (777.000000)

## Certification and provenance

- Certification status: new
- Certification results: UNKNOWN
- Parent matrix parity / row-ID alignment: verified by pre-fit candidate checks
- Locked split signature: 6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47
- Train input fingerprint: {'sha256': '06823ea80cceade018b2428e0c7c9cb89e4efae45fcad8cbf61e8b27b350df20', 'rows': 731942, 'columns': ['transaction_id', 'customer_id', 'timestamp', 'amount_bdt', 'merchant_id', 'merchant_category', 'device_id', 'device_type', 'location', 'payment_method', 'transaction_type', 'account_age_days', 'fraud']}
- Test inference / submission: False / False
- Runtime seconds: 33.678494
- OOF: outputs/oof/R037.csv
- Model/config/spec/report/source: ['outputs/models/R037/fold_1.pkl', 'outputs/models/R037/fold_0.pkl']; outputs/reports/R037//config.json; spec_r037.json; outputs/reports/R037//decision_report.json; outputs/reports/R037//source
- Module hashes: {}
- Runner hashes: {'feature_experiment.py': '929442773a4c0a958be1bd89d0fcdfc0d9ab7d143cfc525dcf96c19dc01cda59', 'react_runner.py': '662ca45b1121307c5160726415d3fcd4a654109328c0d7261a445a5c284bc6e0'}
- Parent cache: {'used': True, 'path': 'outputs\\feature_matrices\\R029.pkl', 'matrix_sha256': 'ab767d22d77a44ef3514e8580d9287be720b210475037158514883c1c249771c', 'manifest_sha256': '433acf180bbe536d3d86d2a2cf559ae2ca4368548cb887ae56a601dbe02a028a'}

## Reference-run analysis

- R020: compatible=False; deltas=UNKNOWN

## Factual conclusion

R037 improved corrected F2 late, July, July 1-7 versus its parent; classification is COMPETITIVE-BLEND-ELIGIBLE and submission gates passed=false.
