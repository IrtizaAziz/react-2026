# REACT 2026 Datathon Rulebook

**Organizer:** IEEE Southeast University (SEU) Student Branch  
**Event:** REACT 2026 Datathon  
**Career Partner:** Aspire Tech

---

## 1. Overview

REACT 2026 Datathon is a live, online Datathon organized by the IEEE Southeast University (SEU) Student Branch, open to undergraduate students nationwide. At launch, registered teams receive a problem statement and dataset; teams train a model, submit predictions, and climb a real-time Kaggle leaderboard scored automatically. The top-ranked teams on the private leaderboard are invited to present onsite at SEU for the final round. The specific problem and dataset stay hidden until launch.

- **Organizer:** IEEE Southeast University (SEU) Student Branch
- **Platform:** Kaggle Community Competition (online)
- **Task:** Revealed live at launch (come with a general-purpose, GPU-ready ML setup)
- **Onsite:** Final presentation + Q&A round at SEU Campus

## 2. Eligibility & Teams

1. Open to currently enrolled undergraduate students from any recognized university in Bangladesh.
2. A valid Student ID card is mandatory for registration and for verification at the venue during the onsite round.
3. Cross-institution teams are allowed — members may come from different universities.
4. **Team size: 2–4 members.** A person may belong to only one team.
5. Each participant needs a verified Kaggle account (phone-verified, required for submissions and, if applicable, GPU access).
6. One team = one Kaggle **"Team"** on the competition (merge individual accounts into a Team on Kaggle before the first submission).
7. Team name must match your registered team name, so organizers can map the leaderboard back to registration.
8. Executive members and core volunteers of the IEEE SEU SB organizing committee are not eligible to participate.
9. Team composition cannot be modified after the registration deadline.

## 3. Schedule

| Milestone | Date / Time |
|---|---|
| Registration | 11–31 August 2026 (Online) |
| Problem & dataset reveal + Online round opens | 06 September 2026, 08:00 AM |
| Online round closes (final Kaggle deadline) | 07 September 2026, 11:59 PM |
| Shortlisted teams' work submission deadline (notebook + summary) | 08 September 2026, 10:00 AM |
| Finalists’ announcement | 08 September 2026 |
| Onsite finals (presentations) | 10 September 2026, SEU Campus |

## 4. Getting onto Kaggle

1. Create and phone-verify your Kaggle account.
2. Open the competition invite link: **[shared privately with each registered team leader at launch]**.
3. Click **Join** and accept the competition rules (mandatory before any submission counts).
4. Form/merge your Team on the competition's **Team** tab before your first submission.

## 5. Data & What You May Use

### At Launch You Will Receive

- A training set.
- A test set (you generate predictions for this).
- Possibly some extra unlabeled data.
- Full sizes, format, and evaluation metric are published with the Kaggle invitation at launch, not before.

### Allowed

- Publicly available pretrained models/backbones, open-source code and libraries, and standard data preprocessing/augmentation techniques.
- Their sources must be disclosed in the method summary (see Section 8), with the model loaded directly inside the submitted notebook.

### Not Allowed

- Manually labeling, or hand-correcting predictions on, the test data.
- Using the test data (or any leaked labels) inside your training/validation loop.
- Using any external dataset not provided by the organizers, or fine-tuning pretrained models on external data.
- Sharing code, data, or predictions across teams (private sharing = collusion).
- Any attempt to de-anonymize or probe the test set to obtain ground truth.

## 6. Submissions (Online Round)

- **Format:** a CSV in the exact layout shown on the competition's Evaluation page and in `sample_submission.csv` (both published at launch).
- Only the prediction CSV is submitted through Kaggle during the online round — no code or report is required at this stage.
- **Submission limit:** 5 per day, per team.
- **Leaderboard split:** the public leaderboard is computed on 60% of the test set; your final rank uses the private leaderboard on the remaining 40%, revealed at close.
- **Final selection:** select up to 2 submissions to count for the private leaderboard (Kaggle default). If none is selected, your best public submission is auto-selected.
- **Do not chase the public leaderboard** — it is a small sample and can mislead. Focus on robust local validation.

## 7. Leaderboard Metric

- Teams are ranked by a single evaluation metric on the held-out test set (**higher is better**), defined precisely on the competition's Evaluation page at launch.
- The exact metric, its tie-break rule, and a worked example are part of the launch reveal.

## 8. Code & Reproducibility (Integrity)

To protect fairness and originality, the top 15 teams on the private leaderboard must submit, by the deadline in Section 3:

- Training + inference code, submitted as a Kaggle Notebook link (see Section 8.1) that reproduces the submitted result.
- A 1–2-page method summary (approach, model, data handling, key results).

### 8.1 Notebook Sharing Requirements

- The notebook must be shared with the organizing committee's Kaggle account(s) as a collaborator (private share) — do not make it fully public, as this could expose your approach to other teams before final ranking is confirmed.
- You must point organizers to the **exact committed notebook version** (via Kaggle's Version History) that generated your leaderboard-scoring `submission.csv` — not simply the current/latest state of the notebook, which may have since been edited.

### 8.2 Consequences

Failure to provide reproducible code, or a result that cannot be reproduced from the submitted notebook version, forfeits the leaderboard position — the onsite slot passes to the next-ranked team.

## 9. Onsite Finals Qualification

- The **top 15 teams by private leaderboard** — subject to passing the reproducibility check in Section 8 — are invited to present onsite.
- Onsite is a live presentation + Q&A session (see Section 11).

## 10. Final Ranking (How the Winner Is Decided)

Final score blends online performance and the onsite pitch using the official REACT 2026 weighting:

- **Online Phase — 60% (Kaggle leaderboard performance):** From the private leaderboard among finalists. Recommended mapping:

  `Online_points = 60 × (team_score − min_score) / (max_score − min_score)`

  across finalists (or rank-based scoring, e.g., 1st = 60, 2nd = 53, …). The chosen method will be stated publicly before the onsite round.

- **Offline Phase — 40% (final presentation & evaluation):** Judges' total score, rescaled to 40 points.
- **Champion = highest Final score;** ties are broken by the higher Kaggle private-leaderboard score.

> **Note:** The organizing committee may adjust it, provided the final formula is published before the online round begins.

## 11. Onsite Presentation Guide

**Format:** 3 minutes, hard stop, with a short Q&A session following the presentation.

Because presentation time is limited, the talk should be self-contained: say the important things, don't save them for Q&A.

### Suggested Structure

- **Problem & framing:** The problem you solved and why it matters.
- **Approach / architecture:** Your model choice and why; your modeling pipeline; any pretrained models used.
- **Key technical decisions:** How you handled data constraints, preprocessing, feature engineering, and validation strategy.
- **Results:** Your leaderboard score and position, plus a quick error/failure analysis.
- **Impact & close:** Real-world relevance and a one-line takeaway.

### Tips

- Practice with a timer — judges may cut you off at the limit. Land your result before the cutoff, not after.
- Lead each slide with the conclusion, then the evidence.
- Show one strong qualitative example of your model's output — it sells more than text.
- Be honest about limitations; rigor is rewarded in judging.
- Bring slides as PDF + PPTX on a USB, and email them to the organizing committee as backup ahead of the onsite round.

## 12. Conduct & Disqualification

Plagiarism, cross-team collusion, test-set manipulation, multiple team registrations by the same individual, or misrepresenting results are grounds for disqualification.

Registration fees are non-refundable for disqualified teams. The organizers' decisions are final and binding.

## 13. Prizes

**Total prize pool: 30,000 BDT**, distributed as:

| Position | Prize |
|---|---:|
| Champions | 50% |
| 1st Runners Up | 30% |
| 2nd Runners Up | 20% |

## Registration Fees

- **Per person:** ৳500
- **Team size:** 2–4
- IEEE Member Discount and Campus Ambassador Referral Discount Available

## Payment

Accepted via **bKash and Nagad** (Personal Account — select **"Send Money"**, not "Payment") to:

**01642963222**

- Send the exact amount based on your event and IEEE status.
- Keep your Transaction ID — it is required during registration.
- Registration is only complete once payment is verified by the organizing committee.
- Reference is compulsory for online payment. Teams have to write:

  `TeamName_Datathon`

  as the reference.

## Inquiries

### Md Maruf Hasan
Chairperson, IEEE SEU SB  
marufhasan@ieee.org  
+88 01642963222

### Salehin Sadek Ayion
Vice Chair, IEEE SEU SB  
salehinsadek@ieee.org  
+88 01889001676

### Salman Sany Jetu
Webmaster, IEEE SEU SB  
salmansanyjetu@gmail.com  
+88 01575332918

## Follow Us On

- **Facebook:** https://www.facebook.com/seu.ieee.sb
- **Instagram:** https://www.instagram.com/ieee.seu.sb
- **LinkedIn:** http://linkedin.com/company/seu-ieee-sb

---

**Source:** Official REACT 2026 Datathon Rulebook.
