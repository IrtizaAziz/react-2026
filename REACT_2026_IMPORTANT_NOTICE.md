# IMPORTANT NOTICE — REACT 2026

**Organizer:** IEEE Southeast University Student Branch  
**Event:** REACT 2026 Datathon  
**Competition Platform:** Kaggle  

> The REACT 2026 competition email has been sent to all registered participants. Since the email may go unnoticed before the competition starts, or may be delivered to the Spam/Junk folder, the complete important information is reproduced here.

Please check your **Inbox** and **Spam/Junk** folder and read the competition email carefully.

---

## 1. Before the Competition

### Kaggle Account

- Every participant must have their own Kaggle account.
- If you do not have one, create an account using an email address you regularly check.
- Every participant must complete **phone verification**.
- You will not be able to submit without a phone-verified account.
- Do not share Kaggle accounts.
- Each participant must use only their own account.
- Using more than one Kaggle account per participant may result in **disqualification**.

---

## 2. Competition Launch

**Date:** 6 September 2026  
**Time:** 8:00 AM (Dhaka Time)

### REACT 2026 Kaggle Contest

https://www.kaggle.com/t/a2ea39a8e900459e88d89983aefd76b4

The competition is **private and invite-only**. Please do not share the invitation link publicly. Only registered participants should access it.

### After Opening the Competition Page

- Click **Join Competition**.
- Accept the competition rules.
- Your submissions will only count after you officially join and accept the rules.
- Carefully read the:
  - Overview
  - Rules
  - Data
  - Evaluation

---

## 3. Team Formation

The **team leader** must create the team from the Competition **Team** tab.

### Important Requirements

- Kaggle team name must **exactly match** the team name used during registration.
- Invite teammates through the Team tab.
- Each teammate must have joined the competition individually before being invited.
- Every teammate must accept the invitation.
- Teams must contain **2–4 members**.
- Team composition cannot be changed, as the registration deadline has already passed.

---

## 4. Online Round

**Opens:** 6 September 2026, 8:00 AM  
**Closes:** 7 September 2026, 11:59 PM (Dhaka Time)

> **This is a hard deadline. Submissions after the deadline will not be counted.**

### Files to Download Once the Competition Opens

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `data_dictionary.csv`

Before starting, carefully read the **Overview** and **Data** tabs.

### Pay Particular Attention To

- Problem statement
- Dataset structure
- Evaluation metric
- Competition constraints
- Permitted approaches
- Validation requirements
- Data leakage restrictions

> **Important:** The dataset contains **time-ordered data**, so the rules regarding feature engineering and validation are especially important.

---

## 5. Model Development

Build your **fraud-detection model** using only the data and resources permitted by the competition rules.

Because the dataset is time-ordered, carefully consider the rules before designing:

- Features
- Feature engineering
- Train/validation split
- Cross-validation strategy
- Model-selection process

Your approach must **not introduce future information or data leakage** into your features or validation process.

---

## 6. Submission Requirements

Your submission must be a CSV file containing **exactly two columns**:

```text
transaction_id
fraud
```

The `fraud` column must contain a **probability between 0 and 1**, not a binary `0/1` prediction.

Use `sample_submission.csv` as the submission template to ensure the correct format and column names.

### Submission Rules

- Maximum **5 submissions per day per team**.
- Each submission will receive a score on the **Public Leaderboard**.
- The Public Leaderboard uses only **60% of the test set**.
- Therefore, do not rely solely on the public leaderboard.
- Your own validation strategy should be the **primary basis for model selection**.

---

## 7. Final Submissions

Before the competition deadline, each team may select **up to 2 submissions** for evaluation on the **Private Leaderboard**.

The **Private Leaderboard score determines the final ranking**.

If you do not select any submissions, your **best-scoring public submission** will be used automatically.

### Final Deadline

**7 September 2026, 11:59 PM (Dhaka Time)**

> Submissions after this deadline will not be accepted under any circumstances.

Please do not wait until the final few minutes to submit your prediction file.

---

## 8. If You Finish in the Top 15

Teams finishing in the **Top 15 on the Private Leaderboard** will advance to:

1. Reproducibility Verification Round
2. Onsite Final at Southeast University (SEU)

### Onsite Final

**10 September 2026**

Detailed instructions regarding the reproducibility verification process, required materials, and submission procedure will be provided directly to qualifying teams after the online round closes and rankings are confirmed.

---

## 9. Important Competition Rules

Please remember:

- No external datasets may be used.
- External fraud-label sources are prohibited.
- Fine-tuning on external data is prohibited.
- Do not share code, data, or predictions with teams outside your own team.
- Attempts to de-anonymize the test data or reverse-engineer how the dataset was generated are strictly prohibited and may be investigated during reproducibility review.
- Plagiarism, collusion, or misrepresentation of results may result in disqualification.
- Registration fees are non-refundable in case of disqualification.

For the complete list of permitted and prohibited practices, refer to the **Overview** and **Rules** tabs on the REACT 2026 competition page.

---

## 10. Questions & Support

### Md Maruf Hasan

**Chairperson, IEEE SEU Student Branch**  
Email: [marufhasan@ieee.org](mailto:marufhasan@ieee.org)

### Salehin Sadek Ayion

**Vice Chair, IEEE SEU Student Branch**  
Email: [salehinsadek@ieee.org](mailto:salehinsadek@ieee.org)

> Competition-related questions should preferably be posted in the **Discussion** tab once the competition opens so that all participating teams can benefit from the same official clarification.

---

## Quick Access

- **Kaggle:** https://www.kaggle.com/
- **Kaggle Account Settings:** https://www.kaggle.com/settings
- **Kaggle Competitions:** https://www.kaggle.com/competitions
- **Kaggle Competition Documentation:** https://www.kaggle.com/docs/competitions

### REACT 2026 Contest

https://www.kaggle.com/t/a2ea39a8e900459e88d89983aefd76b4

---

## Pre-Launch Checklist

Before starting, make sure:

- [ ] Every team member has a Kaggle account.
- [ ] Every Kaggle account is phone-verified.
- [ ] Every member has individually joined the competition.
- [ ] The team leader has created the Kaggle team.
- [ ] Kaggle team name exactly matches the registered team name.
- [ ] All teammates have accepted the team invitation.
- [ ] Competition rules have been accepted.
- [ ] Overview, Rules, Data, and Evaluation tabs have been read.
- [ ] `train.csv` has been downloaded.
- [ ] `test.csv` has been downloaded.
- [ ] `sample_submission.csv` has been downloaded.
- [ ] `data_dictionary.csv` has been downloaded.
- [ ] Validation design accounts for the time-ordered dataset.
- [ ] Submission pipeline outputs exactly `transaction_id` and `fraud`.
- [ ] `fraud` predictions are probabilities in `[0, 1]`.
- [ ] Submission usage is tracked against the 5-per-day limit.
- [ ] Up to 2 strong final submissions are selected before the deadline.

---

**IEEE Southeast University Student Branch**  
**REACT 2026 Organizing Committee**
