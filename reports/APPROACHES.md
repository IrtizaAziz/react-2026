# APPROACHES

Use this page to turn the revealed task into a short, ranked queue. Each item must become a uniquely numbered experiment and use the locked split unless the validation rationale changes.

| Priority | Hypothesis | Controlled change | Gate | Status |
| --- | --- | --- | --- | --- |
| 1 | The official metric and split are implemented exactly | Verify Evaluation worked example and persist validation v1 | Audit + baseline | Ready at launch |
| 2 | Native CatBoost captures interactions missed by linear models | Same features/folds, CatBoost challenger | Full CV + OOF | Accepted in mock as E005 |
| 3 | A genuinely different model reduces OOF error correlation | Add LightGBM/XGBoost or a permitted challenger | Full CV + correlation | Candidate |
| 4 | Entity-derived features generalize | Add one feature family only | Grouped sensitivity, then full CV | Not yet tested |
| 5 | Diverse candidates improve robustly | Explicit OOF blend | Same folds, no public-LB tuning | Candidate |

Matched-fold screening is a cheap kill switch, never a final score. A promising screen must be promoted to full CV before it can influence a submission.
