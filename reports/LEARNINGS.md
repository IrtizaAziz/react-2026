# LEARNINGS

- Keep validation fixed before model search. E001–E006 are comparable because they reuse the same saved folds.
- Fold-local preprocessing is required; one-hot categories must never learn validation-only values.
- E002's row-local `FamilySize` and `IsAlone` improved E001 by 0.007859 CV.
- E005 CatBoost improved E002 by 0.030268 CV and is the selected mock candidate.
- E006 raw `Title` reduced E005 by 0.003365 CV and is rejected on the current path.
- `Ticket` and surname cross the supplied train/test boundary; related features need grouped sensitivity validation.
- Public leaderboard scores are manually recorded evidence, not an optimizer. The E005 CV/LB gap must be treated as a robustness signal.
- AutoML challengers are useful for diversity but their internal validation is non-comparable unless they use our locked folds.
