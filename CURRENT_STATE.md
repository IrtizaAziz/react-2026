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
R007 remains the local standalone recent-period lead. R008's broader velocity family and R009's native one-hot raw merchant identity did not add value; R011 LightGBM is slightly weaker standalone but has moderately different ranks for a later explicitly controlled OOF-blend review. No submission was prepared or uploaded.

Best live CV:
R007: F1 0.75433429; F2 0.69000428; historical S = 0.70930329. R008: F1 0.75539812; F2 0.68970116; S = 0.70941025. R009: F1 0.74293462; F2 0.66854805; S = 0.69086402. R011: F1 0.75030502; F2 0.68834761; S = 0.70193461.

Best live public LB:
None; zero live submissions are recorded for today.

Submissions used today:
0 / 5 live; verify Kaggle's displayed quota/reset boundary before uploading.

Final submissions selected:
None for live competition.

Known leakage risks:
Historical features must use strictly earlier timestamps; equal-timestamp rows cannot update one another. Transaction IDs are chronological identifiers and excluded from features. OOF warmup rows remain unpredicted; fold AP and pooled covered OOF AP are separate quantities.

Current highest-priority task:
Human review of R007 versus the submitted R005 incumbent. R011 is not a standalone submission candidate, but its F2/F1 rank correlations with R007 (Spearman 0.631/0.651) support an explicitly authorized later fixed-weight OOF blend experiment; no blend has been created.

Live experiment namespace:
Use R001, R002, ... for real REACT runs. Preserve E001–E006 unchanged as mock history.
