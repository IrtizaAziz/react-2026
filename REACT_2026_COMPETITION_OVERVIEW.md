# REACT 2026 Datathon — Competition Overview

**Organizer:** IEEE Southeast University (SEU) Student Branch, Bangladesh  
**Competition:** REACT 2026 Datathon  
**Task:** Temporal Fraud Detection  
**Platform:** Kaggle

---

## Overview

REACT 2026 is a national Datathon organized by the IEEE Southeast University (SEU) Student Branch. Teams are given a chronologically ordered stream of raw transaction data and must predict the probability of fraud without being given any behavioral or historical features.

The challenge is to engineer **time-aware, leakage-safe signals** from customer, device, merchant, and location patterns. The **top 15 teams on the private leaderboard** will be invited to the onsite final at Southeast University, subject to reproducibility verification.

---

## Competition Status

- **Start:** 3 hours ago
- **Close:** 2 days to go

---

## Background

Real-world fraud detection is rarely a static classification problem. A transaction is not fraudulent or legitimate in isolation — it is fraudulent or legitimate **relative to a customer's own history**, **relative to the entities involved** (device, merchant, location), and **relative to a moment in time**.

The same ৳30,000 transaction is unremarkable for one customer and a five-alarm anomaly for another. The same device is normal when it belongs to one person and suspicious when it suddenly belongs to twelve.

Fraud detection is also a moving target. Fraud rings adapt. Old signals stop working. New patterns emerge that never existed in your training data. A model that only memorizes what fraud looked like yesterday will miss what fraud looks like tomorrow.

REACT 2026 asks you to build a system that understands **behavior over time** — not just transactions in isolation.

---

## Problem

You are given a large, chronologically ordered stream of transactions from a digital payment ecosystem. Each transaction is described only by raw, unprocessed fields — no behavioral, historical, or aggregated features are provided.

> **Your task: for every transaction in `test.csv`, predict the probability that it is fraudulent.**

You must discover and engineer the behavioral signal yourself: what is normal for this customer, this device, this merchant, this location — and how does this transaction deviate from it?

---

## The Challenge

The columns you are given describe *what happened*. They do not tell you *whether it was normal*. That judgment requires reconstructing, for every transaction, things like:

- **Customer behavior** — how often does this customer transact, for how much, and has that changed recently?
- **Temporal behavior** — is this an unusual hour or day for this customer? Has there been a sudden burst of activity?
- **Device behavior** — has this customer used this device before? Is this device unusually shared across many accounts?
- **Merchant behavior** — is this a merchant the customer knows? Is this merchant currently seeing unusual traffic?
- **Location behavior** — is this a location the customer normally transacts from? Did they just "teleport" from somewhere else?
- **Relationship structure** — do groups of customers, devices, and merchants form suspicious clusters that no single transaction reveals on its own?

None of this is handed to you as a column. Constructing it — correctly, and without looking into the future — **is the competition**.

We will not tell you exactly how fraud was generated in this dataset. Reverse-engineering the generative assumptions is not the intended path to a good score; understanding behavior, generally, is.

---

## Constraints — Read Carefully

This competition is explicitly about **time-aware, leakage-safe modeling**.

### Target Leakage

Using a transaction's own fraud label, or any transaction's future fraud label, to construct a feature is prohibited.

### Temporal Leakage

Computing a "historical" feature using transactions that occur after the transaction being scored is prohibited.

Every engineered feature for a transaction at time `t` may only use information **strictly before `t`**.

### Test-Set Leakage

Do not fit any model, encoder, scaler, or target-related statistic — including target encoding — using `test.csv`.

Feature computation that uses only the raw, non-target columns of `test.csv` in a **strictly-past-only** way is allowed. For example, a device's known transaction history may include test-period rows that occurred earlier than the row being scored.

The `fraud` label must never be used anywhere near `test.csv`, because it does not exist for participants.

### Validation Leakage

Random K-fold cross-validation is **not appropriate** for this task because it allows information from a customer's future transactions to leak into validation scores for earlier transactions.

Use approaches such as:

- Time-based validation
- Walk-forward validation
- Expanding-window validation
- Purged validation
- Embargoed validation

Submissions that appear to rely on validation schemes inconsistent with leaderboard behavior may be asked to reproduce their pipeline.

### Do Not Chase the Public Leaderboard

The public leaderboard uses only **60% of the test set** and is not guaranteed to represent the private 40% well.

Trust a robust, time-aware local validation setup over repeated public submissions.

---

## Evaluation

### Official Metric: PR-AUC (Average Precision)

Fraud is rare — roughly **1.5–2% of transactions**.

Under this level of class imbalance, accuracy and even ROC-AUC can look deceptively strong for a model that mostly predicts "not fraud."

PR-AUC evaluates the tradeoff between precision and recall specifically on the positive fraud class.

Formally:

```text
AP = Σ_n (R_n − R_{n−1}) · P_n
```

where `P_n` and `R_n` are the precision and recall at threshold `n`.

This is exactly:

```python
sklearn.metrics.average_precision_score
```

### Leaderboard

- **60% public**
- **40% private**
- Select up to **2 submissions** for private leaderboard evaluation.
- If none are selected, the best public-scoring submission is used automatically.

---

## Submission Format

Submit a CSV containing exactly these two columns:

```csv
transaction_id,fraud
T000731942,0.0132
T000731943,0.8127
```

`fraud` must be a **probability in `[0, 1]`**, not a hard `0/1` prediction.

---

## Dataset Description

You are provided:

| File | Description |
|---|---|
| `train.csv` | Historical transactions, each labeled `fraud` (`0` or `1`). |
| `test.csv` | Future transactions with no `fraud` column. |
| `sample_submission.csv` | Example submission format. |
| `data_dictionary.csv` | Column-by-column description of raw fields. |

Every transaction in `test.csv` occurs strictly after every transaction in `train.csv`.

At a glance, each transaction includes:

- Transaction ID
- Customer ID
- Timestamp
- Amount (BDT)
- Merchant ID
- Merchant category
- Device ID
- Device type
- Coarse location
- Payment method
- Transaction type
- Customer account age in days at the time of the transaction

A small amount of missingness exists in:

- `merchant_category`
- `device_type`
- `location`

This is intentional.

---

## Dataset Time Range

### `train.csv`

**2026-01-01 00:00:43 → 2026-07-15 23:58:21**

### `test.csv`

**2026-07-16 00:00:21 → 2026-09-15 22:34:38.341114**

Every transaction in `test.csv` occurs strictly after every transaction in `train.csv`.

Do not assume fraud patterns in `test.csv` are identical to those in `train.csv`. Behavioral fraud patterns can drift over time.

---

## What You Must Predict

For each `transaction_id` in `test.csv`, predict:

```text
P(fraud = 1)
```

The value must be a probability in `[0, 1]`.

---

## Important

This dataset intentionally provides **only raw transaction-level fields**. No behavioral, historical, or aggregated features are provided.

Discovering and engineering these features — without leaking future information — is the core challenge of this competition.

---

## Key Takeaways

1. This is a **temporal fraud-detection** problem.
2. The official metric is **PR-AUC / Average Precision**.
3. Fraud prevalence is roughly **1.5–2%**.
4. Random K-fold validation is inappropriate.
5. Historical features must use only information available strictly before each row's timestamp.
6. Raw test-period history may be used only in a strictly past-only, non-target manner.
7. Public leaderboard score should not override trustworthy local validation.
8. Predictions must be probabilities.
9. Public/private split is **60% / 40%**.
10. Top 15 private-leaderboard teams advance subject to reproducibility verification.

---

**Source:** REACT 2026 Kaggle Competition Overview
