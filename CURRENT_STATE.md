# Current state

Competition:
REACT 2026 Datathon

Problem:
Live REACT 2026 fraud detection. E001–E006 remain a closed Titanic rehearsal and are historical evidence only.

Target:
fraud (binary: 0 = nonfraud, 1 = fraud)

Task type:
Binary classification

Official metric:
Average Precision via sklearn.metrics.average_precision_score; higher is better.

Higher/lower better:
higher

Train shape:
731,942 rows x 13 columns

Test shape:
262,648 rows x 12 columns

Submission format:
CSV with 262,648 rows and exactly transaction_id,fraud; fraud is a probability in [0,1].

Validation strategy:
STAGE 1 LOCKED: calendar_time expanding folds. F1 train < 2026-03-14, validate [2026-03-14, 2026-05-15); F2 train < 2026-05-15, validate [2026-05-15, 2026-07-16). See config_live_stage1.json and reports/live_foundation.json.

Best live experiment:
None. Stage 1 contains no training run.

Best live CV:
None.

Best live public LB:
None; zero live submissions are recorded for today.

Submissions used today:
0 / 5 live; verify Kaggle's displayed quota/reset boundary before uploading.

Final submissions selected:
None for live competition.

Known leakage risks:
Historical features must use strictly earlier timestamps; equal-timestamp rows cannot update one another. Transaction IDs are chronological identifiers and excluded from features. OOF warmup rows remain unpredicted; fold AP and pooled covered OOF AP are separate quantities.

Current highest-priority task:
Stage 1 foundation is complete once tests and live metadata are verified. R001 is not yet approved to run; it must use the locked static feature profile and calendar folds.

Live experiment namespace:
Use R001, R002, ... for real REACT runs. Preserve E001–E006 unchanged as mock history.
