# REACT 2026 Datathon

## Competition status and rules

Source: the live Kaggle Competition Overview and official competition rules. They take precedence over older repository assumptions.

- Platform: Kaggle Community Competition.
- Opens: 06 September 2026, 08:00 AM.
- Closes: 07 September 2026, 11:59 PM. The deadline timezone remains unresolved here; verify it on Kaggle.
- Maximum 5 submissions per day per team.
- Public leaderboard: 60% of the test set. Private leaderboard: the remaining 40%.
- Up to 2 submissions may be selected for private scoring. The private leaderboard determines online ranking.
- Top 15 teams are subject to reproducibility verification before advancing onsite.
- No external datasets, external fraud-label sources, or fine-tuning on external data.
- No cross-team sharing of code, data, or predictions.
- No de-anonymization or attempts to reverse-engineer the organizer's fraud-generation process.
- No manual test labeling, hand-corrected predictions, test-label leakage, or test-set probing.
- Public pretrained models/backbones are permitted when loaded in the submitted notebook and disclosed with their exact source/model.
- Top teams must provide reproducible training and inference from the exact committed Kaggle Notebook version that produced the submitted result.

## Task and submission

- Task: binary fraud detection.
- For every row in `test.csv`, predict the probability that `fraud = 1`.
- Submission columns are exactly:

  ```text
  transaction_id,fraud
  ```

- `fraud` must be a probability in `[0,1]`, not a hard class label.

## Official metric

- Metric: PR-AUC / Average Precision.
- Official implementation corresponds to `sklearn.metrics.average_precision_score`.
- Higher is better.
- Fraud prevalence is roughly 1.5–2%.

## Dataset structure

Official files:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `data_dictionary.csv`

Raw transaction fields include, at minimum:

- transaction ID
- customer ID
- timestamp
- amount in BDT
- merchant ID
- merchant category
- device ID
- device type
- coarse location
- payment method
- transaction type
- customer account age in days

Small amounts of missingness exist in `merchant_category`, `device_type`, and `location`.

## Temporal structure

The dataset is explicitly chronological, and every test transaction occurs strictly after every train transaction.

- Train range: `2026-01-01 00:00:43` → `2026-07-15 23:58:21`
- Test range: `2026-07-16 00:00:21` → `2026-09-15 22:34:38.341114`

Fraud behavior may drift over time; do not assume the test distribution is identical to train.

## Leakage and validation constraints

For a transaction at time `t`, historical engineered features may use only information strictly before `t`.

Prohibited:

- target leakage
- future-label leakage
- temporal leakage from future transactions
- fitting models, encoders, scalers, or target-related statistics using `test.csv`
- random K-fold validation

Appropriate validation families include time-based holdout, walk-forward, expanding-window, and purged/embargoed validation.

Raw, non-target information from earlier rows in the test period may be used for strictly past-only sequential features for later test rows. For example, an earlier test-period transaction may become part of a device or customer's known history when scoring a later test transaction. This must never involve target information.

## Core modeling challenge

Only raw transaction-level fields are provided; behavioral, historical, and aggregated features are not supplied. The challenge is to construct leakage-safe historical signals around customers, time/activity patterns, devices, merchants, locations, and relationships between entities.

## Remaining unresolved items

- The official deadline timezone should be verified against Kaggle.
- Exact feature names beyond those explicitly confirmed above should be taken from the released files and `data_dictionary.csv` rather than inferred.
