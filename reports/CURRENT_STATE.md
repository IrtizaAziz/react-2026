# Current state

This file mirrors the root `CURRENT_STATE.md` for tools and future sessions that expect reports to be self-contained. The root file remains the editable source of truth for the rehearsal snapshot.

- Best mock experiment: E005 CatBoost, 0.831643 accuracy CV; manually supplied public LB 0.77511.
- E006 raw Title comparison: rejected at 0.828278 CV.
- E003 and E004: preserved failed CatBoost integration attempts; never reuse their IDs.
- Validation: E001–E006 use the locked five-fold stratified split (seed 42). Entity-derived features require grouped sensitivity checks.
- Highest-value next rehearsal: run the synthetic notebook in a fresh Kaggle runtime, save an exact notebook version, and verify replay.
- Live launch facts: fraud detection, explicitly time-ordered data, expected `train.csv`, `test.csv`, `sample_submission.csv`, and `data_dictionary.csv`.
- Live submission schema: exactly `transaction_id,fraud`; `fraud` must be a probability in `[0,1]`.
- Live unknowns: official metric/direction, timestamp column, target prevalence, train/test continuity, repeated entities, and historical-only feature restrictions.
- Live experiments use `R001`, `R002`, ...; E001–E006 remain immutable mock history.
