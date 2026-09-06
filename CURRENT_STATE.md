# Current state

Competition:
REACT 2026 Datathon

Problem:
Closed mock REACT 2026 launch rehearsal using the supplied Kaggle Titanic files

Target:
Survived (binary: 0 = did not survive, 1 = survived)

Task type:
Binary classification

Official metric:
Mock metric: accuracy

Higher/lower better:
higher

Train shape:
891 rows x 12 columns

Test shape:
418 rows x 11 columns

Submission format:
CSV with 418 rows and columns PassengerId, Survived; PassengerId must align to test.csv

Validation strategy:
ACCEPTED FOR E001 ONLY: StratifiedKFold, 5 folds, shuffle=True, seed=42, accuracy. Any ticket/family/surname-derived feature requires grouped leakage/sensitivity validation.

Best experiment:
E005 (selected best mock experiment)

Best CV:
0.831643 accuracy (E005; 5-fold population standard deviation 0.017827)

Best public LB:
0.77511 (E005; manually supplied)

Submissions used today:
2 / 5

Final submissions selected:
E005 (mock selection)

Known leakage risks:
PassengerId is a sequential train/test split marker (train 1--891; test 892--1309). Name is unique per row. Ticket and surname identify related entities across splits (115 shared tickets; 144 shared surnames). Do not use PassengerId; handle learned preprocessing fold-locally; assess group leakage before using ticket/family-derived features.

Current highest-priority task:
Mock rehearsal is closed: E005 is selected; E006 remains rejected; E003/E004 remain preserved as failed. Live launch facts: fraud detection, time-ordered data, expected files train/test/sample/data_dictionary, exact sample schema transaction_id/fraud with fraud probability in [0,1]. Exact metric, timestamp, entity structure, and final temporal validation remain unknown until Kaggle Evaluation/Data are inspected.

Live experiment namespace:
Use R001, R002, ... for real REACT runs. Preserve E001–E006 unchanged as mock history.
