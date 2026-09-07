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
R017 is the local standalone lead on the locked folds: it appends five strictly-past merchant behavior features to immutable R013 and improves F2 by 0.023294 and July by 0.011229. No submission was prepared or uploaded.

Best live CV:
R017: F1 0.78982897; F2 0.72959419; historical S = 0.74766462. R013: F1 0.76266259; F2 0.70630069; S = 0.72320926. R007: F1 0.75433429; F2 0.69000428; S = 0.70930329. R011: F1 0.75030502; F2 0.68834761; S = 0.70193461.

Best live public LB:
None; zero live submissions are recorded for today.

Submissions used today:
0 / 5 live; verify Kaggle's displayed quota/reset boundary before uploading.

Final submissions selected:
None for live competition.

Known leakage risks:
Historical features must use strictly earlier timestamps; equal-timestamp rows cannot update one another. Transaction IDs are chronological identifiers and excluded from features. OOF warmup rows remain unpredicted; fold AP and pooled covered OOF AP are separate quantities.

Current highest-priority task:
Human review of R017 versus the submitted R013 incumbent. R017 is a standalone candidate only; no submission was prepared or uploaded.

Live experiment namespace:
Use R001, R002, ... for real REACT runs. Preserve E001–E006 unchanged as mock history.
