# Endgame snapshot through R032

Read-only factual snapshot for Astra review. No training, reservation, test inference, prediction generation, submission, or upload was performed for this snapshot.

## Repository state

- Git commit: `fb6887a9609fb6d2406e6c2a9ced35b197ec9a32` (`master`, `origin/master`), subject `Add R013/R017 reproducibility artifacts`.
- Working tree: dirty. The checkout has modified tracked files (`AGENTS.md`, `CURRENT_STATE.md`, `README.md`, `reports/CURRENT_STATE.md`, `src/config.py`, `src/data.py`, `src/device_recent_sharing.py`, `src/train.py`, `experiments/experiments.csv`), deleted raw data files (`data/raw/gender_submission.csv`, `data/raw/test.csv`, `data/raw/train.csv`), and untracked R018–R032/spec/cache/module/test/report artifacts. Exact porcelain output is the source of truth.
- Latest consumed R-ID: `R032`.
- Next valid R-ID: `R033`.

## Experiment table: R017–R032

Scores are AP. `Corrected late` is the documented `[2026-06-15, 2026-07-16)` window; R017's immutable report also contains the older June-15–July-1 diagnostic, which is not used here. `Gate` is the recorded submission-gate result; `—` means no score was produced.

| R-ID | Parent | Controlled change | F1 | F2 | Corrected late | July | July 1–7 | July 8–15 | Classification | Submission gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| R017 | R013 | Add five strictly-past merchant behavior features: prior count, seconds since last, prior unique customers, 24h transactions, 24h unique customers. | 0.789829 | 0.729594 | 0.676524 | 0.509281 | 0.472205 | 0.537056 | WIN | UNKNOWN/not recorded as a gate field |
| R018 | R017 | Replace CatBoost with the fixed R011 LightGBM recipe on R017's exact 50-feature matrix; F2-first screen. | — | — | — | — | — | — | failed | not reached |
| R019 | R017 | Fresh repeat of the same R017-matrix LightGBM comparison after R018 failure. | — | — | — | — | — | — | failed | not reached |
| R020 | R017 | Corrected fresh LightGBM rerun on R017's exact 50-feature matrix. | 0.791093 | 0.730512 | 0.677813 | 0.511806 | 0.481581 | 0.535160 | LOSE | false |
| R021 | R017 | OOF-only fixed blend `0.75*R017 + 0.25*R020`; no retraining or test inference. | 0.795241 | 0.731837 | 0.678391 | 0.511409 | 0.476572 | 0.537625 | FLAT-MIXED / promising but below gate | false |
| R022 | R017 | Add five strictly-past merchant amount-history features. | 0.798206 | 0.726453 | 0.674623 | 0.508689 | 0.472433 | 0.536051 | LOSE | false |
| R023 | R017 | Add five strictly-prior customer new-merchant relationship-creation features. | 0.790210 | 0.728734 | 0.676023 | 0.508563 | 0.470601 | 0.537250 | FLAT-MIXED | false |
| R024 | R017 | Add four strictly-prior device-location familiarity features. | 0.789201 | 0.730074 | 0.677415 | 0.509459 | 0.474059 | 0.536165 | FLAT-MIXED | false |
| R025 | R017 | Add five strictly-past merchant new-customer relationship/composition features. | 0.797973 | 0.730930 | 0.677491 | 0.509857 | 0.474180 | 0.536905 | FLAT-MIXED | false |
| R026 | R017 | Isolate `merchant_new_customer_share_24h` from the merchant new-customer family. | 0.794682 | 0.731790 | 0.678883 | 0.510675 | 0.474082 | 0.538224 | WIN | false |
| R027 | R026 | Isolate `device_location_share_of_device_history` from the device-location family. | 0.797359 | 0.731707 | 0.678816 | 0.511018 | 0.475920 | 0.537728 | FLAT-MIXED | false |
| R028 | R026 | Isolate `customer_seconds_since_last_new_merchant` from the customer new-merchant family. | 0.796804 | 0.731824 | 0.678833 | 0.512080 | 0.476324 | 0.539024 | FLAT-MIXED | false |
| R029 | R026 | Isolate `merchant_seconds_since_last_new_customer` from the merchant new-customer family. | 0.797571 | 0.732476 | 0.679526 | 0.512182 | 0.477828 | 0.538197 | FLAT-MIXED | false |
| R030 | R026 | Attempt to isolate `device_other_customers_24h`; feature certification contract mismatch. | — | — | — | — | — | — | failed — pre-training infrastructure failure | not reached |
| R031 | R026 | Repeat the same `device_other_customers_24h` attempt; raw-frame/derived-input adapter mismatch. | — | — | — | — | — | — | failed — pre-training infrastructure failure | not reached |
| R032 | R026 | Corrected training run isolating `device_other_customers_24h`. | 0.795375 | 0.732881 | 0.679135 | 0.511996 | 0.476164 | 0.539082 | FLAT-MIXED | false |

R030 and R031 are explicitly pre-training infrastructure failures: `training_started=false`, no OOF was created, and no test inference or submission occurred. They are not model losses and have no AP scores.

## Incumbent, parent, manifest, and gates

- Current submitted incumbent: `R017`.
- Locally recorded Public result: AP `0.54460`. Rank evidence conflicts: the strategy opening records rank `22`, while a later policy paragraph refers to rank `18`; rank is therefore not treated as resolved.
- Current development parent under the latest documented policy: `R029`. The policy promotes a completed candidate only when F2 delta >= `+0.0005`, corrected-late delta >= `+0.0005`, July delta >= `+0.0010`, F1 delta >= `−0.0020`, neither July subperiod falls by more than `0.0005`, and causal/feature-parity/replay/fold-ID/provenance checks pass. The document states that R029 is admitted; R032 remains a side candidate.
- Current development-parent feature manifest: R029's 52 ordered features are the R017 50-feature manifest plus `merchant_new_customer_share_24h` and `merchant_seconds_since_last_new_customer`:

  `amount_bdt, log_amount_bdt, account_age_days, hour, weekday, is_weekend, merchant_category, device_type, location, payment_method, transaction_type, merchant_category_missing, device_type_missing, location_missing, customer_prior_count, customer_first_seen, customer_seconds_since_last, customer_observed_age_seconds, customer_prior_mean_amount, customer_prior_mean_log_amount, customer_prior_std_log_amount, customer_amount_to_prior_mean, customer_log_amount_minus_prior_mean, customer_log_amount_zscore, customer_device_prior_count, customer_device_new, customer_device_seconds_since_last, customer_device_count_share, customer_location_prior_count, customer_location_new, customer_location_seconds_since_last, customer_location_count_share, device_prior_count, device_prior_distinct_customer_count, device_prior_distinct_customer_count_excluding_current, device_seconds_since_last, device_observed_age_seconds, device_prior_other_customer_transaction_count, customer_share_of_device_prior_transactions, customer_prior_1h_count, device_prior_1h_count, customer_merchant_prior_count, customer_merchant_is_new, customer_merchant_seconds_since_last, customer_merchant_share_of_customer_history, merchant_prior_transaction_count, merchant_seconds_since_last, merchant_prior_unique_customers, merchant_transactions_24h, merchant_unique_customers_24h, merchant_new_customer_share_24h, merchant_seconds_since_last_new_customer`.

- Current submission gate, measured against R017: July >= `+0.003`, corrected late >= `+0.002`, F2 >= `+0.003`, F1 no worse than `−0.002`, and each July subperiod no worse than `−0.002`; also required are causal/parity/fold-ID/replay checks, no tiny-subgroup-only gain, exact final-fit reproduction, schema verification, and explicit human authorization. No R018–R032 candidate has a recorded `true` submission-gate result.
- Repeatedly improving but not promoted as a feature child: the isolated salvage features in R027, R028, R029, and R032 all improve some recent or F2 diagnostics relative to R026, but only R029 is recorded as meeting the documented development-parent promotion policy. `merchant_new_customer_share_24h` (R026) is the accepted feature WIN; the other listed features were not all promoted into a further completed child manifest.

## Salvage experiments

| R-ID | Feature isolated | Recorded result |
|---|---|---|
| R022 | Merchant amount-history family: five merchant-conditioned amount statistics | LOSE; F1 improved, recent/F2 evidence declined |
| R023 | Customer new-merchant relationship-creation family: five features | FLAT-MIXED |
| R024 | Device-location familiarity family: four features | FLAT-MIXED |
| R025 | Merchant new-customer relationship/composition family: five features | FLAT-MIXED |
| R026 | `merchant_new_customer_share_24h` | WIN; accepted feature isolation |
| R027 | `device_location_share_of_device_history` | FLAT-MIXED |
| R028 | `customer_seconds_since_last_new_merchant` | FLAT-MIXED |
| R029 | `merchant_seconds_since_last_new_customer` | FLAT-MIXED; admitted as development parent by policy |
| R030 | `device_other_customers_24h` | Pre-training infrastructure failure; no feature result |
| R031 | `device_other_customers_24h` | Pre-training infrastructure failure; no feature result |
| R032 | `device_other_customers_24h` | FLAT-MIXED |

## Model diversity: R020/R021

- R020 is the corrected LightGBM rerun using R017's exact 50-feature causal matrix. It recorded F1 `0.791093`, F2 `0.730512`, corrected late `0.677813`, July `0.511806`, July 1–7 `0.481581`, and July 8–15 `0.535160`; its classification is `LOSE` because July 8–15 fell by more than the R020 screen allowed.
- R021 is the fixed OOF-only blend `0.75*R017 + 0.25*R020`. It recorded F1 `0.795241`, F2 `0.731837`, corrected late `0.678391`, July `0.511409`, July 1–7 `0.476572`, and July 8–15 `0.537625`. It improved the main R017 F1/F2/late/July values by approximately `+0.005412`, `+0.002242`, `+0.001867`, and `+0.002128`, but did not clear the strict submission floors; no test predictions or submission were created.

## Parent-matrix caches and verification

| Cache | State | Evidence |
|---|---|---|
| R017 | Present and provenance-recorded | `outputs/feature_matrices/R017.json` and `.pkl`; 50 columns; split signature `6fb939...`; matrix SHA-256 `e71957...`; frozen-source provenance present. |
| R026 | Present and provenance-recorded | `outputs/feature_matrices/R026.json` and `.pkl`; 51 columns; split signature `6fb939...`; matrix SHA-256 `54881b...`; R026 report records the parent-cache verification as true. |
| R029 | No dedicated cache found | R029 report/config exists and is complete, but no `outputs/feature_matrices/R029.*` was found. |

R027–R032 provenance records reference the R026 parent cache hash `54881bb19b4d9c306fc3731247a8b9ba3dc10170541d2e86b3f8018dbf454887` and record parent-cache verification true where applicable. Generic provenance rows for R017–R025 remain `UNKNOWN` in `reports/EXPERIMENT_INDEX.md` even though the R017/R026 cache artifacts exist.

## Certified generic feature modules

The locally available generic modules and their declared `AVAILABLE_FEATURES` are:

- `src.merchant_new_customers`: `merchant_new_customers_24h`, `merchant_new_customers_7d`, `merchant_new_customer_share_24h`, `merchant_new_customer_share_7d`, `merchant_seconds_since_last_new_customer`. Certification evidence is present; R026/R029 use certified source lineage.
- `src.customer_new_merchants`: `customer_new_merchants_7d`, `customer_new_merchants_30d`, `customer_new_merchant_share_7d`, `customer_new_merchant_share_30d`, `customer_seconds_since_last_new_merchant`. Certification source exists; R028's provenance marks the selected source as not certified in the index.
- `src.device_location`: `device_location_prior_count`, `device_location_is_new`, `device_location_seconds_since_last`, `device_location_share_of_device_history`. Certification source exists; R027's provenance marks the selected source as not certified in the index.
- `src.device_recent_sharing`: `device_unique_customers_24h`, `device_unique_customers_7d`, `device_other_customers_24h`, `device_other_customers_7d`, `device_7d_unique_to_lifetime_unique_ratio`. R030/R031 failed before training; R032's provenance marks the selected source as not certified in the index.

## Reproducibility state and blockers

- Locked calendar folds and split signature are recorded and reused: `6fb939b019dfe071ce44a3b6e2c82c87bdf17aafaa5dec714c1b703e77cd7e47`.
- R017/R026 reports include train fingerprints, frozen source hashes, feature manifests, replay records, and causal/parity assertions. R017's fold replays are recorded verified.
- The checkout is dirty, and several run records were made from dirty states; no clean source-to-commit mapping for every later artifact is recorded.
- The strategy records unresolved recovery/verification blockers for original R013/R017 OOF/fold/final artifacts and exact committed Kaggle notebook versions, plus a mismatch between generic export semantics and the all-train final-refit path.
- The local submission log is blank/stale for live submissions; exact Kaggle submission IDs/timestamps and quota/reset state are not locally resolved.

## Deadline

- Exact competition close recorded in the strategy: **7 September 2026, 23:59 Dhaka time**.
- The rulebook also records **8 September 2026, 10:00 AM** for shortlisted-team notebook and 1–2-page summary delivery.
- The local `COMPETITION.md` still says the deadline timezone requires verification. The strategy's 21:00 modeling stop and 22:00 submission target are internal targets, not the competition close.

## Previous Astra plan coverage

The exact project-source document named `REACT FINAL PLAN` was not found. The classifications below are taken from the locally present `REACT_2026_ENDGAME_STRATEGY.md`, especially its “Champion ideas not yet used” list, and cite only materially matching R-IDs; related experiments are not silently treated as equivalent.

| Proposed feature/model mechanism | Status | R-ID evidence |
|---|---|---|
| LightGBM on current best/R017 features | TESTED | R018, R019 failures; R020 completed |
| XGBoost on current best features | UNTESTED | No R-ID |
| Fixed blend of current models | PARTIALLY TESTED | R016 and R021 are fixed blends, but neither is an equal-weight current-model blend |
| Merchant-relative amount/dispersion | PARTIALLY TESTED | R022 tested merchant amount history, not the exact proposed mechanism |
| Customer–merchant amount baseline | UNTESTED | R022 is merchant-conditioned, not customer–merchant |
| Customer median/IQR/percentile amount features | UNTESTED | No R-ID |
| Category/payment/type-conditioned amounts | UNTESTED | No R-ID |
| Relationship turnover/new-edge rate | PARTIALLY TESTED | R023 and R025 tested specific new-merchant/new-customer families, not the full turnover-rate mechanism |
| Customer rolling merchant/device/location breadth | PARTIALLY TESTED | R008 tested customer/device windows; R015 tested recent device sharing; exact proposed breadth package not isolated |
| Customer category/payment/type breadth | UNTESTED | No R-ID |
| Device merchant/location breadth and new-customer proportion | PARTIALLY TESTED | R024 tested device-location familiarity; R026 tested merchant new-customer proportion; exact combined breadth mechanism not tested |
| Device–location familiarity | TESTED | R024; R027 isolated its share component |
| Device–merchant and merchant–location familiarity | UNTESTED | No materially matching R-ID |
| Non-overlapping acceleration | PARTIALLY TESTED | R008 tested 1h/7d counts and ratios, not the exact non-overlap mechanism |
| Personalized time-bin/weekday habits | UNTESTED | No R-ID |
| Previous-batch category/switch summaries | UNTESTED | No R-ID |
| Time-bucket occupancy/entropy/burstiness | UNTESTED | No R-ID |
| Additional frequency encodings | PARTIALLY TESTED | R003–R008, R013, R017 cover major customer/device/merchant/pair counts; no distinct location-frequency ablation |
| Target/WOE encodings, neighbor fraud scores, label propagation | EXCLUDED/UNSAFE | No R-ID; excluded by documented label-timing/leakage policy |
| Full-period UID smoothing, future/reverse counts | EXCLUDED/UNSAFE | No R-ID; future information/identity-inference concern |
| Pseudo-labeling and train-vs-test classifier | EXCLUDED/UNSAFE | No R-ID; excluded by current rules |
| Learned stacking/optimized blend weights | PARTIALLY TESTED | R016 and R021 fixed arithmetic blends only; no learned weights or weight search |
| Centralities, communities, embeddings, GNNs | UNTESTED | No R-ID |
| Transformer/RNN/CNN histories and nested supervised models | UNTESTED | No R-ID |
| Hashing, SMOTE, broad class-weight search | UNTESTED | No R-ID |
| Calibration and hard thresholds | UNTESTED | No R-ID; no ranking experiment recorded |

## Evidence unresolved

- No exact file named `REACT FINAL PLAN` exists in the inspected project/workspace; the coverage section uses `REACT_2026_ENDGAME_STRATEGY.md` as the available plan source.
- Public rank is conflicting locally: rank 22 is recorded in the strategy opening and rank 18 in a later policy paragraph. Public AP `0.54460` is recorded.
- Exact Kaggle notebook version, live submission identifiers/timestamps, quota/reset state, and original R013/R017 submitted-byte provenance are not resolved locally.
- R029 has no dedicated parent-matrix cache artifact; generic provenance for several earlier IDs remains `UNKNOWN` in the experiment index.
