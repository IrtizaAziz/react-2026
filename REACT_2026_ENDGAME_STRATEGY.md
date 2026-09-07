# REACT 2026 endgame strategy

## A. Executive decision

- **Recommended phase: MODEL DIVERSITY / ENSEMBLE**, with reproduction readiness as a prerequisite.
- R017 is the strongest **recorded** candidate: it improves every locked decision window over R013.
- Your supplied Public score is **0.54460**, rank **22**; the current #1 score is **0.56548**.
- The gap to your approximate first-place score, 0.566, is **0.02140**.
- The highest-value next model experiment is **LightGBM on R017’s exact 50 features**.
- Immediately recover and verify the R017/R013 candidate packages: essential models, OOF files, and submission CSVs are absent from this checkout.
- Recommend **R017 primary + R013 hedge**, subject to exact submission/version verification.
- Deadline: **7 September, 23:59 Dhaka time**; another submission slot before then is **UNKNOWN**.
- No model was trained, experiment ID reserved, new test prediction generated, artifact modified, or Kaggle action taken during this review.

## B. Verified current state

### Evidence and its limits

| Item | Finding |
|---|---|
| Official task | Temporal fraud probability ranking; `sklearn.metrics.average_precision_score` |
| Inputs | 731,942 training rows; 262,648 test rows |
| Input integrity | Train, test, and sample-submission SHA256 values match recorded fingerprints |
| Training prevalence | **1.7605%** |
| Training entities | 38,689 customers; 19,089 devices; 4,178 merchants |
| Training duplicates | No duplicate transaction IDs; no exact duplicate rows after excluding transaction ID |
| Equal timestamps | **40,466 training rows** belong to tied timestamp groups |
| Validation | Locked expanding F1/F2 calendar folds; unchanged |
| F1 | Train before 14 March; validate 14 March–14 May |
| F2 | Train before 15 May; validate 15 May–15 July |
| July diagnostic | 59,465 rows; **926 positives** |
| R017 | 50 features; CatBoost, 800 iterations, depth 6, seed 42 |
| R013 | 45 features; same CatBoost recipe |
| Frozen source integrity | R013/R017 source snapshots and final-package copies match their recorded hashes |
| Recent OOF/models | R006–R017 OOF files and recent fold models are missing locally |
| Final artifacts | R013/R017 test prediction files exist and match metadata hashes; final models and submission CSVs are missing locally |
| Notebook readiness | No R013/R017 reproduction notebook or exact committed Kaggle version was found |
| Submission history | Scores come from your supplied request; the local submission log is blank |
| Current checkout | Clean at `4aa582e`; R013/R017 runs record `3f82097` with a dirty working tree |

The state summaries contain outdated claims about submissions and experiment progress. They must not override actual artifacts or your manually supplied Public results.

The [official notice](D:/React-2026/REACT_2026_IMPORTANT_NOTICE.md:69) explicitly resolves the closing timezone. The [rulebook](D:/React-2026/REACT_2026_DATATHON_RULEBOOK.md:37) specifies **8 September, 10:00 AM** for shortlisted teams’ notebook and 1–2-page summary.

The competition invitation reached a Kaggle sign-in page. Consequently, current rank, #15 score, remaining quota, reset time, and existing private selections could not be independently verified.

### Checks performed during this review

- Independently recomputed R003/R004/R005 F2, late, and July AP from available OOF; results match the reports.
- Calculated strictly-past entity/pair support and existing merchant-feature correlations from training data.
- Checked existing merchant-engine oracle parity, tied-row permutation, prefix independence, and future-value mutation independence using a small deterministic feature-only check.
- Inspected saved test predictions for **label-free score diversity**, without fitting or inferring labels.

R013/R017 AP, replay, and subgroup claims remain **recorded evidence**, not newly reproduced measurements, because their OOF/models are unavailable here.

## C. R001–R017 experiment map

“Accepted” below describes the established strategy lineage. The ledger’s execution status is generally `completed`; its conclusion fields remain blank.

| Run | Parent | Hypothesis / controlled change | F2 AP | July AP | Result and lesson |
|---|---|---|---:|---:|---|
| R001 | — | Static CatBoost baseline | — | — | **Failed:** feature-importance name extraction error |
| R002 | R001 | Repair artifact capture; same baseline | .315387 | .209339 | **Accepted baseline** |
| R003 | R002 | Customer history and relative amount | .574879 | .416282 | **Accepted:** F2 +.259492; July +.206943 |
| R004 | R003 | Customer–device/location familiarity | .619012 | .456092 | **Accepted:** F2 +.044133; July +.039810 |
| R005 | R004 | Device-global support and sharing | .678318 | .488139 | **Accepted:** F2 +.059306; July +.032047 |
| R006 | R005 | Training weights with 60-day half-life | .676258 | .487858 | **Rejected:** F2 −.002060; no recent improvement |
| R007 | R005 | Customer/device 1-hour counts | .690004 | .495294 | **Accepted:** F2 +.011686; July +.007155 |
| R008 | R007 | Customer/device 24h/7d counts and ratios | .689701 | .495070 | **Flat/mixed; not promoted:** broader tested package adds little |
| R009 | R007 | Raw merchant ID with target-independent one-hot | .668548 | .475079 | **Rejected:** F2 −.021456; July −.020215 |
| R010 | R007 | LightGBM with identical features | — | — | **Failed:** feature-importance name extraction error |
| R011 | R007 | Repair LightGBM importance labeling | .688348 | .496050 | **Not promoted standalone:** F2 −.001657; July +.000756; diversity candidate |
| R012 | R007 | Six 30-day customer amount features | .688297 | .494087 | **Flat/mixed:** F2 −.001707; July −.001207 |
| R013 | R007 | Customer–merchant familiarity | .706301 | .498052 | **Accepted WIN:** F2 +.016296; July +.002758 |
| R014 | R013 | Customer–category familiarity | .706630 | .499174 | **Flat/mixed:** F2 +.000330; July +.001122 |
| R015 | R013 | Recent distinct-customer device sharing | .707625 | .496892 | **Flat:** F2 +.001324; July −.001160 |
| R016 | R013 | .75 R013 + .25 R011 probability blend | .706864 | .500003 | **Flat/mixed:** F2 +.000563; late −.000169 |
| R017 | R013 | Five merchant history features | **.729594** | **.509281** | **Accepted WIN:** F2 +.023294; July +.011229 |

The successful lineage is:

**R002 → R003 → R004 → R005 → R007 → R013 → R017**

R006, R008, R009, and R012–R016 are not components of R017.

## D. Champion × REACT transfer matrix

### Research corrections

The attached report is useful as an idea inventory, but unreliable as a ready-to-run strategy:

- Its bracketed references lack a usable bibliography.
- “All winners used strictly past-only features” is unsupported.
- **GroupKFold over month groups does not by itself enforce forward-only training.**
- Plain grouped `diff()` can let equal-timestamp rows see each other.
- “Train-only” aggregation can still leak later training events into earlier rows.
- CatBoost ordered statistics and ordinary OOF target encoding do not establish REACT legality.
- Threshold tuning does not improve continuous-ranking AP. Strictly increasing calibration preserves rankings; isotonic calibration can introduce ties. [Official AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- Pseudo-labeling does not inherently reveal true labels, but fitting on test-derived pseudo-labels conflicts with REACT’s test-fitting prohibition.
- Claims about exact AmEx uplift and unspecified Home Credit winners should not support our decisions.

The strongest verified external evidence is:

| Source | Technique and purpose | REACT transfer |
|---|---|---|
| IEEE-CIS first-place account | Entity aggregates and a reconstructed client identifier contextualized raw transactions; heterogeneous boosting combined models | Existing IDs already provide explicit entities. Transfer past-only behavioral summaries and model diversity; exclude identity reconstruction aimed at de-anonymization and full-period prediction smoothing. [NVIDIA account](https://developer.nvidia.com/blog/?p=23421) |
| TalkingData solution repository | Counts, cumulative counts, ratios, unique counts, time buckets, variance, WOE, and next-click features represented activity | Transfer the behavioral questions. Reimplement counts and gaps with strict timestamp batching; exclude reverse counts, next-click features, and unsafe WOE. [Original repository](https://github.com/CuteChibiko/TalkingData) |
| AmEx solution evidence | Historical summaries, interactions, and diverse models represented longitudinal customer behavior | Useful analogy for past-relative change; different target granularity and metric limit transfer. The attached report’s exact 15th-place claims remain weakly sourced. [Participant repository](https://github.com/JEddy92/amex_default_kaggle) |
| Home Credit fifth-place account | Historical aggregation, LightGBM, neural and nested models summarized multiple histories | Supports history/context modeling, but its many tables and months of work differ substantially from REACT. Expensive nested/deep approaches have poor deadline value here. [Team account](https://deepsense.ai/blog/wait-so-loans-need-to-be-repaid-the-home-credit-risk-prediction-competition-on-kaggle/) |

### Complete family reconciliation

These classifications are REACT judgments, not claims that every technique won elsewhere.

| Family | Current classification | Remaining potential and recommendation |
|---|---|---|
| **A. Multiscale activity** | **ALREADY PROVEN / PARTIALLY EXPLORED** | Customer/device 1h wins; their tested 24h/7d package is flat. Merchant 24h is accepted. Non-overlapping acceleration remains distinct, but no window sweep |
| **B. Recent vs long-term change** | **PARTIALLY EXPLORED** | R012 rejects one customer-amount formulation. Relationship turnover and merchant composition change remain untested |
| **C. Familiarity** | **ALREADY PROVEN** | Customer–device/location/merchant win. Customer–category is mostly saturated. Device–location is the best-supported unused pair |
| **D. Turnover/switching** | **GENUINELY UNTESTED** | Recent creation of relationships differs from current-pair novelty; promising but more implementation risk |
| **E. Rolling diversity** | **PARTIALLY EXPLORED** | Device/customer degree and merchant customer breadth partly covered. Customer breadth and merchant device breadth remain open |
| **F. Frequency/count encoding** | **ALREADY PROVEN / REDUNDANT** | Past counts already encode customer/device/merchant/pair frequency. Renaming them as frequency features adds no evidence; location history remains untested |
| **G. Recency** | **ALREADY PROVEN / PARTIALLY EXPLORED** | Main entity and accepted pair recencies exist. Payment/type/device–location gaps require a specific mechanism |
| **H. Amount behavior** | **ALREADY PROVEN / HIGH-VALUE REMAINING** | Customer-relative amount is strong. Merchant-relative amount is distinct; exact customer quantiles and pair quantiles are costlier or sparse |
| **I. Merchant system** | **ALREADY PROVEN / PARTIALLY EXPLORED** | Popularity/support is strongly represented. Merchant-relative amounts or customer composition are more distinct than extra traffic counts |
| **J. Simple graph projections** | **ALREADY PROVEN / PARTIALLY EXPLORED** | Device customer degree is present. Customer-side breadth and device–location context remain plausible |
| **K. Communities/centrality** | **LOW ROI NEAR DEADLINE** | Causal components are possible, but hubs may dominate; centrality, embeddings, and GNNs add large correctness/runtime burdens. Exclude label propagation |
| **L. Personalized time habits** | **GENUINELY UNTESTED** | Repeat history supports coarse habits for many customers; hour/weekday global features already exist. Defer fine bins and recent habit shifts |
| **M. Sequences** | **GENUINELY UNTESTED / LOW ROI** | Simple prior-batch category/switch summaries are plausible. A “previous row” needs an order-independent tie policy. Defer neural sequence models |
| **N. Target/WOE encoding** | **ILLEGAL / UNSAFE under current contract** | Label-availability timing is unproven; chronological OOF alone is insufficient. Exclude |
| **O. CatBoost CTRs** | **UNSAFE if enabled without proof** | Current low-cardinality categories fit below the configured one-hot threshold. Preserve target-independent handling; verify fitted metadata |
| **P. Model diversity** | **HIGH-VALUE REMAINING** | LightGBM(R017) is untested and inexpensive on the originating runtime. XGBoost is a lower-priority challenger |
| **Q. Ensembles** | **PARTIALLY EXPLORED** | R016 tested one old-feature blend. One fixed blend with a current-feature challenger remains justified; avoid weight search and stacking |
| **R. Calibration/thresholding** | **REDUNDANT / LOW ROI** | No threshold optimization for AP. Calibration only for a separately justified probability-use objective |
| **S. Pseudo-labeling** | **ILLEGAL / UNSAFE** | Exclude test-derived fitting |
| **T. Future-event features** | **ILLEGAL** | Exclude next-event, reverse cumulative, future aggregates and future prediction pooling |
| **U. Train-vs-test classifier** | **PROHIBITED** | Use descriptive distribution checks; no classifier fitted using test membership |
| **V. Sampling/class weighting** | **LOW ROI** | R006 concerns recency weighting, not all imbalance methods. No demonstrated issue justifies SMOTE, downsampling, or class-weight search |
| **W. Hashing/dimensionality** | **LOW ROI** | No current dimensionality bottleneck; do not resurrect failed raw merchant identity through hashing |
| **X. Deep learning** | **LOW ROI NEAR DEADLINE** | No local evidence justifies new training, sequence construction, and replay infrastructure |

## E. Feature coverage matrix

Legend: **A** accepted in R017; **F** tested flat/mixed; **R** rejected; **N** never tested. All proposed historical cells require strict-past implementation. Full-period or label-derived versions are **unsafe**, regardless of cell.

### Counts, timing, amounts, breadth

| Axis | Lifetime count | Recent count | Recency | Amount baseline | Robust amount | Unique breadth |
|---|---|---|---|---|---|---|
| Customer | A | A:1h; F:24h/7d | A | A:lifetime; F:30d | N | N |
| Device | A | A:1h; F:24h/7d | A | N | N | A:customers; F:recent customers; N:merchants/locations |
| Merchant | A | A:24h | A | N | N | A:customers/lifetime+24h; N:devices/locations |
| Location | N | N | N | N | N | N |
| Customer × device | A | N | A | N | N | N |
| Customer × location | A | N | A | N | N | N |
| Customer × merchant | A | N | A | N:limited support | N:limited support | N |
| Customer × category | F | N | F | N | N | N |
| Device × merchant | N | N | N | N:limited support | N:limited support | N |
| Device × location | N | N | N | N | N | N |
| Merchant × location | N | N | N | N | N | N |
| Customer × payment | N | N | N | N | N:limited support | N |
| Customer × type | N | N | N | N | N:limited support | N |

### Relationships, change and identity

| Axis | Novelty | Pair share | Short/long change | Turnover | Personal time habit | Static identity / category context |
|---|---|---|---|---|---|---|
| Customer | A:first seen | — | F:activity/amount packages | N | N | Raw ID excluded; global time A |
| Device | A:zero support implicit | A:customer share | F:activity/recent sharing | N | N | Raw ID excluded; device type A |
| Merchant | A:zero support implicit | N:merchant-side share | N:explicit change | N | N | Raw merchant ID R; category A |
| Location | N | — | N | N | N | Location category A |
| Customer × device | A | A | N | N | N | Pair ID excluded |
| Customer × location | A | A | N | N | N | Location A |
| Customer × merchant | A | A:customer denominator | N | N | N | Merchant category A |
| Customer × category | F | F | N | N | N | Category A |
| Device × merchant | N | N | N | N | N | Merchant category A |
| Device × location | N | N | N | N | N | Location A |
| Merchant × location | N | N | N | N | N | Location/category A |
| Customer × payment | N | N | N | N | N | Payment category A |
| Customer × type | N | N | N | N | N | Transaction type A |

**Support prevents a Cartesian sweep:**

- In July, **97.0%** of transactions have at least 20 prior merchant events.
- Only **14.2%** have at least 20 prior customer–merchant events; **16.6%** are new pairs.
- Device–location has at least five prior events for **79.6%** of July rows.
- Device–merchant reaches that support for **53.2%**.
- Merchant–location reaches it for **94.0%**, but may duplicate merchant popularity and location.
- Customer payment/type pairs reach five prior events for **64.4% / 67.3%**.
- Customer July history has a median of **33 prior transactions**: enough for coarse habits, often insufficient for finely divided history.

Customer merchant/device/location/category/payment/type diversity is untested. Merchant device/location breadth and device merchant/location breadth are also untested. Lifetime breadth, recent breadth, breadth-per-transaction, and turnover are different hypotheses; none should be silently bundled.

## F. What has actually driven gains

Ranked by controlled F2 improvements:

1. **Customer-relative amount/history:** +.259492.
2. **Device-global context:** +.059306.
3. **Customer–device/location familiarity:** +.044133.
4. **Merchant behavioral context:** +.023294.
5. **Customer–merchant familiarity:** +.016296.
6. **Immediate activity bursts:** +.011686.

These establish predictive value for feature families. They do not establish the organizer’s fraud-generation mechanism or prove a causal explanation.

The strongest narrative is supported: **raw transactions become informative when expressed relative to past behavior and relationships.**

## G. What has failed or saturated

- **Raw merchant identity:** substantially worse across recent diagnostics. Behavioral merchant state later improved ranking.
- **More customer/device windows:** the tested R008 package was redundant; this does not reject non-overlapping acceleration.
- **Recent customer amount averages:** R012 added no useful recent signal beyond lifetime baselines.
- **Broader category familiarity:** R014 had small improvements, consistent with redundancy after exact merchant familiarity.
- **More recent device sharing:** R015 found striking subgroups but did not improve July. High subgroup fraud prevalence alone is not sufficient.
- **Old-feature blending:** R016’s tiny F2 gain and slightly negative late result do not justify promotion.
- **Recency weighting:** R006 supplies evidence against blindly shortening the effective training history.

A useful warning from R015: extremely high fraud prevalence in a small sharing subgroup can coexist with **no useful aggregate improvement**. Do not turn such observations into hand-coded prediction rules or attempts to infer the data generator.

## H. Why R017 won

### Recorded importance

| Added feature | F2 importance | F1 importance |
|---|---:|---:|
| Prior merchant transaction count | 5.475 | 3.343 |
| Merchant recency | 3.014 | 2.922 |
| Prior unique customers | 3.030 | 2.042 |
| Merchant transactions, 24h | .960 | 1.269 |
| Merchant unique customers, 24h | 1.460 | 1.416 |
| **Combined** | **13.94** | **10.99** |

These are model importance values, not additive shares of AP improvement.

### New read-only evidence

Across F2, Spearman correlations are approximately:

- Lifetime count versus lifetime customer breadth: **1.00**, rounded.
- Lifetime count versus 24h traffic/breadth: **.99**.
- 24h traffic versus 24h customer breadth: **1.00**, rounded.
- Merchant recency versus lifetime count: **−.84**.

The features largely describe overlapping merchant scale/activity.

F2 fraud prevalence by strictly-prior merchant support:

| Prior events | Rows | Positives | Fraud prevalence |
|---|---:|---:|---:|
| 0 | 193 | 22 | 11.40% |
| 1–4 | 1,505 | 137 | 9.10% |
| 5–19 | 6,129 | 523 | 8.53% |
| 20–99 | 13,922 | 519 | 3.73% |
| 100–499 | 18,701 | 328 | 1.75% |
| 500+ | 199,615 | 2,499 | 1.25% |

This supports **merchant support/popularity as a substantial predictive axis**. The 24h prevalence pattern is not simply “more traffic means more fraud.”

There is also substantial concentration: although merchants have only 10 transactions at the entity-level median over the full training period, the typical F2 transaction occurs at a merchant with thousands of prior events. Entity-weighted and transaction-weighted descriptions differ sharply.

### Conclusion

R017 probably captured a major merchant support mechanism that was missing from R013. Evidence for an entire untapped merchant traffic system is weaker:

- Most added features are highly correlated.
- Lifetime count has the largest importance.
- The largest period gain occurs in June 15–30, although both July subperiods improve.
- Additional counts may merely reproduce the same ordering.

**Merchant-relative amount and customer composition remain distinct possibilities. More merchant traffic windows are not the default next step.**

Unavailable OOF prevents attributing R017’s AP gain to merchant-support groups or measuring its interaction with pair novelty. Even with those files, subgroup comparisons would remain descriptive; isolating feature contributions would require separately authorized ablations.

### Late-window correction

The correct late interval is **[15 June, 16 July)**, with AP **.676524**. The immutable R017 diagnostic config incorrectly ends that named window on 1 July, producing **.810133**, which is actually June 15–30.

Preserve the original record. Link every comparison and reproduction package to the [corrected decision report](D:/React-2026/outputs/reports/R017/decision_report.json). Do not silently rewrite the historical config.

## I. Remaining high-upside mechanisms

Ranked by plausible information beyond R017, before accounting for implementation cost:

1. **Amount abnormality conditional on merchant.** Strong history support; distinct from customer amount and merchant counts. Risk: merchant mix makes a global merchant baseline noisy.
2. **Sudden expansion of customer relationships.** Several newly encountered merchants/devices over a recent period can differ from a single novel pair.
3. **Unexpected device–location context.** Strong repeat support and a clear mechanism; may duplicate customer-location familiarity.
4. **Merchant customer-composition change.** New-customer proportion or device breadth could separate traffic volume from who generates it.
5. **Personal temporal-habit violation.** Coarse customer time-bin familiarity is supported for many rows; fine-grained sequences are not.
6. **Non-overlapping activity acceleration.** Distinct from R008’s tested ratios, but current count collinearity reduces expected return.

These are research reserves, not authorization for six experiments.

## J. Model-diversity analysis

### Should we train LightGBM on R017 features?

**Yes—once the candidate artifacts and a path to an eligible submission are secured.**

| Consideration | Assessment |
|---|---|
| Standalone expectation | Could be competitive; no basis to promise it will exceed R017 |
| Local evidence | R011 was only .001657 below R007 on F2 and slightly ahead in July |
| Missing information in R011 | Four customer–merchant features and all five merchant features |
| Diversity evidence | R016 reports Pearson .9726 and rank correlation .7030 between R013/R011; recent error diversity may exist |
| Cost | Existing LightGBM adapter and preprocessing; no new history engine |
| Runtime | R011 recorded about 117 seconds for training; R017 feature generation adds roughly minutes on the originating machine |
| Main risk | Same stronger features make LightGBM track CatBoost without complementary errors |
| Categorical treatment | Keep R011’s fold-local, target-independent one-hot preprocessing; five low-cardinality categories do not justify a new encoding strategy |
| Current environment | The inspected bundled runtime lacks sklearn/scipy; use the established training environment, not an improvised replacement |

A concrete implementation issue exists: the current R017 parity assertion requires the parent’s CatBoost parameters. A LightGBM configuration cannot simply be launched unchanged. Add an explicit model-comparison contract that verifies the **entire R017 feature matrix**, while preserving R017’s historical checks.

CatBoost’s numeric-then-categorical preprocessing explains categorical indices 45–49 in its saved parameters; those are not the five merchant numeric features. CatBoost documents that one-hot categories do not receive CTRs. Verify the actual fitted representation when models are recovered. [CatBoost documentation](https://catboost.ai/docs/en/features/categorical-features)

### Alternatives

- **XGBoost(R017):** plausible but lower priority; no local challenger result and additional runtime/dependency verification.
- **RF/ExtraTrees:** no evidence of useful recent error diversity; sparse categoricals and weaker rare-event ranking are concerns.
- **Logistic model:** inexpensive but likely misses established nonlinear interactions.
- **Recent-window/regime models:** could help drift but discard labels or introduce routing decisions; R006 discourages blind recency searches.
- **Stacking/optimized weights:** insufficient independent temporal evidence for added fitting and tuning.

For expected gain per implementation hour, **LightGBM(R017) outranks another feature family**.

## K. Validation ↔ Public-LB analysis

Public scores below are manually supplied. Local scores come from saved reports, with R003–R005 independently recomputed.

| Run | Model parent | F1 | F2 | Correct late | July | Public |
|---|---|---:|---:|---:|---:|---:|
| R003 | R002 | .667109 | .574879 | .546923 | .416282 | .43058 |
| R004 | R003 | .709915 | .619012 | .578687 | .456092 | .47898 |
| R005 | R004 | .748580 | .678318 | .617453 | .488139 | .50762 |
| R013 | R007 | .762663 | .706301 | .649944 | .498052 | .52256 |
| R017 | R013 | .789829 | .729594 | .676524 | .509281 | .54460 |

The next table compares **successive submissions**, not necessarily direct model parents:

| Transition | ΔF1 | ΔF2 | Δlate | ΔJuly | ΔPublic |
|---|---:|---:|---:|---:|---:|
| R003→R004 | +.042806 | +.044133 | +.031763 | +.039810 | **+.04840** |
| R004→R005 | +.038665 | +.059306 | +.038766 | +.032047 | **+.02864** |
| R005→R013 | +.014082 | +.027982 | +.032491 | +.009913 | **+.01494** |
| R013→R017 | +.027166 | +.023294 | +.026580 | +.011229 | **+.02204** |

Total R003→R017 Public improvement: **+.11402**.

**Interpretation:**

- All four transitions improve every listed local metric and Public AP.
- Validation has been directionally credible for substantial improvements.
- Five selected submissions cannot identify a uniquely “best-correlated” diagnostic or quantify qualification probability.
- Public AP is consistently below full F2, but above July. Full F2 is optimistic as an absolute forecast; July is a stress test, not a calibrated Public prediction.
- July is closest to the test boundary, but contains only 926 positives and has been repeatedly consulted.
- Retain **July → late → F2 → F1** while requiring agreement across windows; do not treat July as untouched confirmation data.

The public/private partition is not documented as chronological. Do not equate Public with early test and Private with late test.

## L. Phase decision

**MODEL DIVERSITY / ENSEMBLE.**

This choice does not assert that all feature engineering is exhausted. It reflects:

- strong coverage of the mechanisms already producing large gains;
- diminishing returns from adjacent count/category/sharing packages;
- substantial redundancy within merchant activity features;
- an inexpensive, unanswered model-family comparison using the complete best feature set.

Execution remains gated by artifact recovery, runtime readiness, and verified submission opportunity. If these prerequisites fail, stop modeling and preserve the current candidates.

## M. Top remaining experiments

Recommend **two**, conditionally. Do not allocate IDs during planning.

### 1. LightGBM on the exact R017 feature set — next experiment

- **Hypothesis:** LightGBM can exploit the strongest relationship and merchant context differently enough to improve recent ranking or provide useful diversity.
- **Parent:** R017 for features and comparison; R011 for the fixed model/preprocessing recipe.
- **Exact change:** Replace CatBoost with R011’s LightGBM recipe; retain all 50 R017 input features, order, raw-event semantics, folds, seed, labels and unweighted AP.
- **Parameters:** 800 estimators, learning rate .05, 31 leaves, minimum child samples 100, L2 5, column fraction .9, row fraction .8, subsample frequency 1, seed 42, eight threads, deterministic and column-wise settings as recorded in R011.
- **Distinctness:** R011 never saw R013/R017’s nine added features.
- **Champion evidence:** Heterogeneous boosting is demonstrated in IEEE-CIS; this supplies a hypothesis, not a forecast.
- **Expected July/late/F2 effect:** Direction uncertain; the desired outcome is consistent improvement or sufficiently competitive recent predictions for one fixed blend.
- **Implementation estimate:** 45–90 minutes on the established environment, including the model-specific parity contract and validation.
- **Runtime estimate:** Approximately 5–15 minutes for features, fits and replay on comparable hardware; budget 30 minutes.
- **Causal complexity:** Low: no new historical family.
- **Redundancy risk:** Medium/high.
- **Probable failure:** More flexible partitions add variance or simply reproduce CatBoost’s errors.
- **Stopping criterion:** Run F2 first. If July or late falls more than .005, or full F2 falls more than .010, stop this challenger. Otherwise complete replay/F1 before deciding whether it merits the single blend test. These are screening loss limits, not promotion criteria.

### 2. One fixed probability blend

- **Hypothesis:** A modest LightGBM contribution corrects some R017 ranking errors.
- **Parent:** R017; members are R017 and the completed experiment above.
- **Exact change:** **.75 × R017 + .25 × LightGBM**, using aligned immutable OOF probabilities.
- **Mechanism:** Combine differing rankings without fitting a meta-model.
- **Distinctness:** R016 combined R013 with older-feature R011.
- **Champion/local evidence:** Heterogeneous ensembles are plausible; R016 also demonstrates that their gain can be too small to matter.
- **Expected July/late/F2 effect:** Small positive or flat is more plausible than another R017-sized jump; no numerical uplift is promised.
- **Implementation estimate:** 15–30 minutes.
- **Runtime:** Seconds for arithmetic; minutes for integrity checks and diagnostics.
- **Causal complexity:** Low for OOF; final inference must preserve each member’s train-only fit and identical causal histories.
- **Redundancy risk:** High.
- **Probable failure:** Dilution by a weaker model.
- **Stopping criterion:** Evaluate once against the submission gates below. No weight grid, rank-blend follow-up, or stacking rescue.

Merchant-relative amount is the highest-ranked **deferred feature experiment**. It would require a new strategic approval rather than becoming an automatic third run.

## N. Next action

**Recover and verify the R017/R013 candidate packages from the originating training machine.**

Completion means locating the original OOF, fold models, final models, actual submitted CSVs, submission identifiers, and any committed notebook versions; verifying their available hashes; and identifying a reproducible training runtime.

This comes before another model because a stronger score without a valid reproduction chain cannot secure qualification.

## O. Conditional next three actions

1. **If recovery succeeds:** establish dedicated R017/R013 notebook reproduction paths and verify Kaggle quota/reset and deadline availability.
2. **If reproduction is ready and a slot can be used before close:** run the single LightGBM(R017) experiment after explicit authorization.
3. **Then branch:** a competitive LightGBM result permits the one fixed blend; a flat result without useful recent diversity ends modeling; a clear loss, replay failure, unavailable quota, or unresolved artifact blocker ends modeling immediately.

If recovery is partial, focus on completing the primary candidate first. If original submitted bytes cannot be found, do not claim that a newly reconstructed CSV is the historically submitted artifact.

## P. Submission strategy

### Quota

Five known historical submissions do **not** establish that today’s quota is exhausted. Their exact Kaggle timestamps and the reset boundary are unavailable.

- Remaining slots: **UNKNOWN**.
- Next reset: **UNKNOWN**.
- Another slot before close: **UNKNOWN**.
- Do not infer reset at local midnight or UTC midnight.

Read the actual submission/quota display before committing modeling time to a new Public candidate. If no slot can become available before close, further candidate development cannot improve this competition’s submitted pool.

### Development-parent promotion vs submission gate

R017 remains the submitted incumbent. R026 remains the last clear feature WIN, but the endgame children show that development-parent promotion and submission selection need different standards. Against R026, R027 improved F1/July but lost F2; R028 improved F1/F2/July but lost corrected late; R029 improved every principal window and both July subperiods; R032 improved F1/F2/late/July and both July subperiods, but its corrected-late gain was only +.000252. The corresponding R029 deltas were F1 +.002890, F2 +.000686, corrected late +.000643, July +.001507, July 1–7 +.003746, and July 8–15 −.000027.

Use the following two-level policy, with deltas measured against the current development parent unless stated otherwise.

**Development-parent promotion (feature accumulation):** promote a completed candidate when all of these hold:

- F2 delta >= **+.0005**;
- corrected F2-late delta >= **+.0005**;
- July 1–15 delta >= **+.0010**;
- F1 delta >= **−.0020**;
- neither July subperiod drops by more than **.0005**; and
- causal, feature-parity, replay, fold/ID, and provenance checks pass.

This is a consistency rule, not a claim of statistical significance. It admits R029 as the new development parent. It does not admit R027 or R028. R032 remains a side candidate because it clears F2 and broad July checks but misses the corrected-late promotion floor.

**July/F2 risk:** repeated selection on July and F2 can overfit the only recent labeled window, especially July’s 926 positives and the July 1–7 slice. R029’s +.003746 July 1–7 gain but essentially zero July 8–15 change is a warning against treating a local early-July win as broad transfer. Merchant/customer composition features can also drift in the unseen test horizon. Use July/F2 as diagnostics for consistency, not as a reason to keep lowering thresholds or to discard the incumbent without the strict submission gate.

**Submission gate (unchanged and strict, against R017):** a candidate must clear every operational floor below, plus exact final-fit reproduction, submission-schema verification, and explicit human submission authorization. Development-parent status never authorizes a submission.

For a scarce new slot, retain these **operational submission floors** against R017:

| Diagnostic | Required |
|---|---:|
| July AP | At least **+.003** |
| Correct F2-late AP | At least **+.002** |
| Full F2 AP | At least **+.003** |
| F1 AP | No loss worse than **−.002** |
| Each July subperiod | No loss worse than **−.002** |

These floors separate a candidate from the observed flat/mixed range; they are not statistical significance thresholds. Additionally require:

- complete causal, feature-parity, fold/ID and replay checks;
- gains not obviously attributable to a tiny merchant/time subgroup;
- exact final-fit reproduction and submission schema verification;
- explicit human submission instruction.

Do not relax these gates because a candidate is architecturally interesting or Public rank remains 18.

Use **21:00 Dhaka** as the proposed modeling stop, **22:00** as the target for completing any approved submission, leaving the remaining time for verification and human selections. Stop earlier if packaging needs more time.

## Q. Private-LB selection

Recommend exactly:

| Candidate | Reason to select | Main private risk |
|---|---|---|
| **R017 — primary** | Best recorded F1/F2/late/July; improves both July subperiods; strong Public corroboration | Merchant support relationships may shift; much lower July than June performance |
| **R013 — hedge** | Strongest submitted candidate without merchant-global features; materially better recent metrics than R005 | Shares most features/model structure with R017 and may simply remain weaker |

Existing R013/R017 test predictions show:

- Pearson correlation: **.971719**
- Spearman rank correlation: **.812626**
- Top-1,000 overlap: **82.5%**
- Top-5,000 overlap: **88.06%**

These establish some prediction differences, **not complementary correctness**. OOF error diversity remains unverified.

R005 is a more primitive hedge, with July .488139 and late .617453 versus R013’s .498052/.649944. There is insufficient evidence to sacrifice that quality for hypothetical diversity.

Replace R013 only with an **actually submitted, reproducible** candidate offering stronger recent performance and useful OOF diversity. Replace R017 only if another candidate broadly exceeds it or its integrity/reproduction fails. Selections remain human actions.

Public rank can change materially on the unseen 40%; its direction and magnitude are unknowable here. No defensible numerical Top-15 probability follows from rank 18 alone.

## R. Reproducibility blockers

| Urgency | Finding | Required resolution |
|---|---|---|
| **CRITICAL NOW** | R013/R017 OOF, fold models, final models and submitted CSVs are absent locally | Recover originals and verify hashes; preserve both candidates |
| **CRITICAL NOW** | No exact committed Kaggle notebook version found | Establish whether one already produced each submission; obtain the exact version reference |
| **CRITICAL NOW** | Generic export follows CV/fold inference, while submitted candidates use all-train refits | Build dedicated final-refit reproduction notebooks |
| **CRITICAL NOW** | No current training environment available through inspected runtime | Locate the originating environment; preserve recorded versions |
| **FIX BEFORE DEADLINE** | Submission log/counters and state summaries are stale | Record manually verified submission IDs, times, scores and selections without rewriting experiments |
| **FIX BEFORE DEADLINE** | R017 named late window is incorrect in its immutable config | Preserve original plus clearly linked correction; prevent automated readers using .810133 as full late AP |
| **FIX BEFORE DEADLINE** | Runs record a dirty Git state | Preserve source hashes and final launcher; create an explicit source-to-commit mapping |
| **FIX BEFORE DEADLINE** | Chunk test concatenates all chunks first | Describe it accurately; do not claim serialized streaming-state equivalence |
| **FIX IF TOP 15** | Organizer sharing and delivery unverified | Privately share the exact version and deliver summary by the official deadline |
| **NICE TO HAVE** | Presentation material absent | Generate the ranked figures after candidate readiness |

### Specific source mismatches

The [generic exporter](D:/React-2026/src/export_notebook.py:104) invokes mean fold-model inference when applicable. R013/R017 were exported through **all-labeled-train final fits**. Their original CV records also have `predict_test=false`, so generic export alone does not establish the submission path.

The [merchant chunk wrapper](D:/React-2026/src/merchant_history.py:43) concatenates chunks before invoking the whole-stream engine. That verifies concatenation behavior, not persistence/resumption of feature state. R017’s final path uses a combined chronological raw stream; retain that proven execution style rather than introducing a streaming rewrite near deadline.

Recorded submission hashes to verify against recovered files:

- **R017:** `ab0e85def7218f8d7331d83c5c61ebf93e47c5596949ad0e0d6969d78ac7bd1f`
- **R013:** `96ea2b8867cc8a53a3db4a43f62d183a73843b0471c56d935bc454c8060f94a9`

A matching prediction-file hash is useful but does not establish which CSV was uploaded.

## S. Finalist deliverable plan

### Reproduction notebook

For each recommended candidate:

1. Embed or faithfully reconstruct its frozen source, exact config and final-fit launcher.
2. Map only organizer inputs and verify fingerprints.
3. Recreate strictly-past training features and fit preprocessing/model on labeled training rows only.
4. Build chronological raw-event history, isolate equal timestamps, restore test transaction order, and perform the same all-train final-fit inference.
5. Compare predictions with the saved reference and verify the exact submission schema and serialization.
6. Record runtime, seeds, feature order, source hashes, split signature, submission hash and notebook version.

The exact source environment records Python 3.14.4, CatBoost 1.2.10, pandas 3.0.5, numpy 2.5.2, sklearn 1.9.0, and LightGBM 4.7.0. These are recorded provenance, not versions newly validated here. Cross-platform equality must be tested.

The rulebook asks for the exact committed version that generated the scoring CSV. A later notebook that merely produces similar scores does not automatically satisfy that wording. If the submission was generated only locally, preserve that fact and resolve the version requirement without misrepresenting its origin.

### Verification cases

- Strict `< t`, equal-timestamp isolation and permutation invariance.
- Future-row and future-value independence.
- Label independence of history state.
- Oracle parity and exact transaction alignment.
- Window boundaries, missing categories, new entities and zero-history behavior.
- Corrected late-window AP and unchanged fold signature.
- Fold-local learned preprocessing.
- Exact final-refit semantics, raw-history continuation and sample-submission order.
- Fresh notebook execution and comparison against recovered reference artifacts.

### Method summary

Prepare 1–2 pages covering task/metric, temporal validation, feature lineage, causal implementation, model recipe, actual CV/Public results, failed approaches, limitations and provenance. No external datasets or pretrained models were used in these candidates.

The two team members should work together on one verified package and narrative.

## T. Final-round narrative

**Problem:** Rank rare fraudulent transactions in a changing chronological stream.

**Insight:** Amounts, devices and merchants become meaningful relative to prior customer behavior and relationships.

**Method:** Timestamp-batched, raw-event historical features plus conservative gradient-boosted trees; expanding temporal validation.

**Evidence:** Customer amount/history → relationship familiarity → device context → merchant familiarity → merchant context. Adjacent ideas were rejected when recent evidence failed.

**Result:** R017 recorded F2 .729594 and July .509281; manually supplied Public AP .54460. Private outcome remains unknown.

**Why it works:** The features replace opaque identifiers with supported descriptions of familiarity, scale, recency and deviation.

### Strongest presentation insights

1. Behavioral merchant history helped substantially after raw merchant ID hurt.
2. Customer–merchant familiarity added signal beyond customer/device/location context.
3. Equal-timestamp isolation is essential with over 40,000 tied training rows.
4. Recent validation prevented promotion of small, inconsistent mean-CV improvements.
5. More windows, recent averages and ensembling were not automatically beneficial.

### Figures to generate later, ranked by judge value

1. **Experiment progression:** controlled F2/July gains with accepted and rejected branches.
2. **Causal architecture:** read prior state → score timestamp batch → update raw state.
3. **Time robustness:** early, late and July comparisons for R005/R013/R017.
4. **Local/Public progression:** all five submitted candidates with clear provenance labels.
5. **Merchant evidence:** support/prevalence plus importance and redundancy, explicitly descriptive.

### Likely judge questions

| Question | Answer direction |
|---|---|
| Why CatBoost? | Strong established nonlinear baseline; conservative fixed recipe; empirical progression, not categorical CTR reliance |
| Why not random K-fold? | It violates the future deployment setting and mixes entity histories across time |
| How was leakage prevented? | Strict prior timestamps, batch isolation, raw-only history and fold-local learned preprocessing |
| Why may earlier test events update history? | Official rules explicitly permit raw, non-target past test events |
| Why did raw merchant ID hurt? | Identity memorization failed locally; support and behavior generalized better in recorded windows |
| Why did R013/R017 work? | They added relationship and merchant context; importance does not prove causal attribution |
| Why not deep learning? | No local evidence justified its development/replay cost before deadline |
| Why not target encoding? | Event-time label availability and valid chronological semantics were not established |
| How robust is the model to drift? | It improved multiple windows, but July performance exposes substantial remaining weakness |
| What failed? | Raw merchant identity, recency weighting and several adjacent history packages |
| Can the exact score be reproduced? | Answer from the committed notebook and artifact comparison only; current checkout is not yet sufficient |

### Adversarial check of seven beliefs

| Belief | Supporting evidence | Contrary evidence | Conclusion / confidence |
|---|---|---|---|
| R017 is strongest | Wins every recorded window and Public comparison | Missing recent OOF/models prevent independent replay | Strongest recorded candidate; **high relative, incomplete verification** |
| July deserves priority | Closest to test boundary; harder regime | 926 positives, repeated selection, shorter horizon | Retain priority with broader checks; **moderate** |
| Merchant expansion remains promising | R017 wins; amount/composition absent | Current count/breadth correlations near one | Distinct merchant mechanisms only; **moderate-low** |
| LightGBM(R017) could diversify | R011 competitive; R016 showed rank differences | R016 gain tiny; stronger features may reduce diversity | One bounded test justified; **moderate** |
| R013 is the right hedge | Strong next submitted candidate; excludes merchant-global state | Closely correlated; no verified OOF error complementarity | Best current hedge recommendation; **moderate** |
| Validation predicts leaderboard | Four consistently positive transitions | Five selected points; absolute optimism; private unseen | Useful directionally, not a rank forecaster; **moderate** |
| No major champion idea was missed | Main count/relationship/amount mechanisms covered | Turnover, conditional amount and habits remain open | Coverage is broad, not exhaustive; **moderate** |

## U. Champion ideas not yet used

| Unused idea | Disposition |
|---|---|
| LightGBM on current best features | **Still valuable:** next model experiment |
| XGBoost on current best features | **Deferred:** behind existing LightGBM infrastructure |
| Fixed blend of equally current models | **Still valuable:** conditional second experiment |
| Merchant-relative amount/dispersion | **Still valuable:** strongest deferred feature family |
| Customer–merchant amount baseline | **Deferred:** sparse for many rows; require explicit support policy |
| Customer median/IQR/percentile | **Deferred:** distinct but more state/correctness cost than means |
| Category/payment/type-conditioned amounts | **Deferred:** possible context, uncertain increment |
| Relationship turnover/new-edge rate | **Still valuable:** coherent missing mechanism, higher causal complexity |
| Customer rolling merchant/device/location breadth | **Deferred:** distinct from R015, potentially useful |
| Customer category/payment/type breadth | **Deferred:** lower-cardinality saturation risk |
| Device merchant/location breadth | **Deferred:** possible overlap with existing sharing/familiarity |
| Merchant device/location breadth and new-customer proportion | **Deferred:** distinct composition signal; volume redundancy must be checked |
| Device–location familiarity | **Still plausible:** strongest-supported unused pair |
| Device–merchant/merchant–location familiarity | **Deferred:** support exists, increment uncertain |
| Non-overlapping acceleration | **Deferred:** not disproved by R008, but no window sweep |
| Personalized time-bin/weekday habits | **Deferred:** coarse histories first if revisited |
| Previous-batch category/switch summaries | **Deferred:** explicit equal-timestamp semantics required |
| Time-bucket occupancy/entropy/burstiness | **Deferred:** distinct summaries, unproven recent gain |
| Additional frequency encodings | **Mostly redundant:** existing prior counts cover main entities |
| Target/WOE, neighbor fraud scores, label propagation | **Excluded:** label-timing or test-fitting concerns |
| Full-period UID smoothing, future/reverse counts | **Excluded:** future information and/or forbidden identity inference |
| Pseudo-labeling / train-vs-test classifier | **Excluded by current rules** |
| Learned stacking / optimized blend weights | **Deferred:** selection-overfit risk and insufficient independent temporal evidence |
| Centralities, communities, embeddings, GNNs | **Too expensive near deadline** |
| Transformer/RNN/CNN histories and nested supervised models | **Too expensive near deadline** |
| Hashing, SMOTE, broad class-weight search | **Low ROI:** no demonstrated problem they solve |
| Calibration and hard thresholds | **No ranking justification for current AP objective** |

## V. Stopping rule

Stop modeling at the first of:

1. No verified route to submit an eligible new candidate before close.
2. R017/R013 provenance or reproduction remains unresolved.
3. The LightGBM challenger clearly loses the predefined recent/F2 screen.
4. The single fixed blend fails the submission standard.
5. Candidate improvement is confined to the observed flat/mixed range.
6. New work requires unsafe features, new validation, broad tuning, or an unfamiliar modeling system.
7. **21:00 Dhaka**, or earlier if reproduction and submission need the remaining time.

The authorized future modeling budget should be **one challenger and, conditionally, one fixed blend**. Further structural discovery requires a separate evidence-based decision. Preserve **R017 + R013** while completing the exact reproduction chain.
