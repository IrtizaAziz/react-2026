# REACT 2026 Datathon

## Confirmed rules

Source: the supplied brief and the local organizer rulebook, ../REACT_2026_DATATHON_RULEBOOK.md. Check launch materials for updates and record any differences explicitly.

- Platform: Kaggle Community Competition.
- Opens: 06 September 2026, 08:00 AM.
- Closes: 07 September 2026, 11:59 PM.
- Deadline timezone: not explicitly stated in the supplied text; verify against the Kaggle deadline.
- Maximum five submissions per day per team. Daily reset boundary: verify on Kaggle.
- Public leaderboard: 60% of test data; private: remaining 40%.
- Online qualification depends on private ranking. Up to two submissions may be selected for private scoring. Selection is a human action.
- External datasets not supplied by organizers are prohibited. No external-data fine-tuning.
- Public pretrained models/backbones are permitted; load them directly inside the submitted notebook and disclose source/exact model in the method summary.
- No test labels, leaked labels, test-set probing/de-anonymization, manual labeling, or hand-corrected predictions.
- Test data cannot enter training/validation; legitimate schema/distribution analysis and inference are permitted.
- No private cross-team sharing of code, data, or predictions.
- Top teams must provide reproducible training and inference in an exact committed Kaggle Notebook version that generated the submitted result. Share privately with the organizers.
- The local rulebook specifies the top 15 teams and a notebook plus 1–2-page method summary deadline of 08 September 2026, 10:00 AM.
- Each participant needs a phone-verified Kaggle account. Form the competition Team before its first submission, using the registered team name; the invitation arrives at launch.
- Rulebook wording says the leaderboard metric is “higher is better.” The implementation intentionally leaves direction unknown until the actual Evaluation definition is checked, as instructed in the brief.

## Revealed problem information

- Target: UNKNOWN
- Task: UNKNOWN
- Metric: UNKNOWN
- Metric direction verified against Evaluation: UNKNOWN
- Metric implementation and official worked-example check: UNKNOWN
- Submission format: UNKNOWN
- Dataset structure: UNKNOWN
- Special restrictions: UNKNOWN
- Competition URL / Evaluation URL: UNKNOWN

## Reasonable assumptions

## Strategy decisions

No competition strategy selected. Record dated evidence, alternatives, and the team's decision here after launch. Baseline model adapters and validation helpers are available capabilities, not approved strategies.

## Implementation decisions already approved

- Local modular Python source, plus a self-contained notebook exporter.
- Tiny synthetic integration fit/replay for isolated smoke testing.
- Descriptive drift checks only; no fitting with test data.
- No automatic Kaggle submission or final selection.
