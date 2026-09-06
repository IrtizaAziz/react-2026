# Data audit

Generated: 2026-09-04T10:40:18.902935+00:00

Diagnostics only. No features are removed, no model is fitted, and no test labels are inferred.
Default sample cap: 100000; mode: FULL; seed: 42.
Counts and distributions below refer to analyzed rows unless explicitly called exact file totals.
JSON, Parquet, and Excel readers may load a complete table before sampling; CSV/TSV/JSONL use bounded samples.

## File inventory
- data\raw\gender_submission.csv: 3,258 bytes
- data\raw\test.csv: 28,629 bytes
- data\raw\train.csv: 61,194 bytes

## gender_submission.csv
Exact file rows: 418; columns: 2; analyzed rows: 418. COMPLETE TABLE.
Analyzed memory: 6,820 bytes.
Columns: ['PassengerId', 'Survived']
Candidate IDs/keys (name heuristic): []
Exact duplicate rows in analyzed data: 0

| Column | Type | Missing count (%) | Unique (ratio) | Signals |
|---|---|---|---|---|
| PassengerId | int64 | 0 (0.00%) | 418 (1.000) | likely numeric/continuous; confirm semantics, near-unique; possible identifier |
| Survived | int64 | 0 (0.00%) | 2 (0.005) | likely categorical |

Top missingness masks (1=missing, column order above): {'00': 418}
Duplicate candidate-feature rows excluding named ID candidates and any configured target: 0; confirm feature roles before interpretation.
- Numeric PassengerId: {'count': 418.0, 'mean': 1100.5, 'std': 120.81045760473994, 'min': 892.0, '1%': 896.17, '5%': 912.85, '25%': 996.25, '50%': 1100.5, '75%': 1204.75, '95%': 1288.15, '99%': 1304.83, 'max': 1309.0}; skew=0.0; 3-IQR extremes=0; nonfinite=0.
- Numeric Survived: {'count': 418.0, 'mean': 0.36363636363636365, 'std': 0.4816221409322309, 'min': 0.0, '1%': 0.0, '5%': 0.0, '25%': 0.0, '50%': 0.0, '75%': 1.0, '95%': 1.0, '99%': 1.0, 'max': 1.0}; skew=0.5689905804746247; 3-IQR extremes=0; nonfinite=0.

## test.csv
Exact file rows: 418; columns: 11; analyzed rows: 418. COMPLETE TABLE.
Analyzed memory: 134,174 bytes.
Columns: ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
Candidate IDs/keys (name heuristic): []
Exact duplicate rows in analyzed data: 0

| Column | Type | Missing count (%) | Unique (ratio) | Signals |
|---|---|---|---|---|
| PassengerId | int64 | 0 (0.00%) | 418 (1.000) | likely numeric/continuous; confirm semantics, near-unique; possible identifier |
| Pclass | int64 | 0 (0.00%) | 3 (0.007) | likely categorical |
| Name | str | 0 (0.00%) | 418 (1.000) | likely categorical, high cardinality, near-unique; possible identifier |
| Sex | str | 0 (0.00%) | 2 (0.005) | likely categorical |
| Age | float64 | 86 (20.57%) | 79 (0.189) | likely numeric/continuous; confirm semantics |
| SibSp | int64 | 0 (0.00%) | 7 (0.017) | likely categorical |
| Parch | int64 | 0 (0.00%) | 8 (0.019) | likely categorical |
| Ticket | str | 0 (0.00%) | 363 (0.868) | likely categorical, high cardinality |
| Fare | float64 | 1 (0.24%) | 169 (0.404) | likely numeric/continuous; confirm semantics |
| Cabin | str | 327 (78.23%) | 76 (0.182) | likely categorical |
| Embarked | str | 0 (0.00%) | 3 (0.007) | likely categorical |

Top missingness masks (1=missing, column order above): {'00000000010': 244, '00000000000': 87, '00001000010': 82, '00001000000': 4, '00000000110': 1}
Duplicate candidate-feature rows excluding named ID candidates and any configured target: 0; confirm feature roles before interpretation.
- Numeric PassengerId: {'count': 418.0, 'mean': 1100.5, 'std': 120.81045760473994, 'min': 892.0, '1%': 896.17, '5%': 912.85, '25%': 996.25, '50%': 1100.5, '75%': 1204.75, '95%': 1288.15, '99%': 1304.83, 'max': 1309.0}; skew=0.0; 3-IQR extremes=0; nonfinite=0.
- Numeric Pclass: {'count': 418.0, 'mean': 2.2655502392344498, 'std': 0.8418375519640503, 'min': 1.0, '1%': 1.0, '5%': 1.0, '25%': 1.0, '50%': 3.0, '75%': 3.0, '95%': 3.0, '99%': 3.0, 'max': 3.0}; skew=-0.5341703482345055; 3-IQR extremes=0; nonfinite=0.
- Categorical Name: common={'Kelly, Mr. James': 1, 'Wilkes, Mrs. James (Ellen Needs)': 1, 'Myles, Mr. Thomas Francis': 1, 'Wirz, Mr. Albert': 1, 'Hirvonen, Mrs. Alexander (Helga E Lindqvist)': 1, 'Svensson, ; singleton values=418; rare examples={'Kelly, Mr. James': 1, 'Wilkes, Mrs. James (Ellen Needs)': 1, 'Myles, Mr. Thomas Francis': 1, 'Wirz, Mr. Albert': 1, 'Hirvonen, Mrs. Alexander (Helga E Lindqvist)': 1, 'Svensson, .
- Categorical Sex: common={'male': 266, 'female': 152}; singleton values=0; rare examples={}.
- Numeric Age: {'count': 332.0, 'mean': 30.272590361445783, 'std': 14.181209235624422, 'min': 0.17, '1%': 0.8579, '5%': 8.0, '25%': 21.0, '50%': 27.0, '75%': 39.0, '95%': 57.0, '99%': 64.0, 'max': 76.0}; skew=0.4573612871503845; 3-IQR extremes=0; nonfinite=0.
- Numeric SibSp: {'count': 418.0, 'mean': 0.4473684210526316, 'std': 0.8967595611217135, 'min': 0.0, '1%': 0.0, '5%': 0.0, '25%': 0.0, '50%': 0.0, '75%': 1.0, '95%': 2.0, '99%': 4.0, 'max': 8.0}; skew=4.168336568562722; 3-IQR extremes=3; nonfinite=0.
- Numeric Parch: {'count': 418.0, 'mean': 0.3923444976076555, 'std': 0.9814288785371691, 'min': 0.0, '1%': 0.0, '5%': 0.0, '25%': 0.0, '50%': 0.0, '75%': 0.0, '95%': 2.0, '99%': 4.0, 'max': 9.0}; skew=4.654461698299236; 3-IQR extremes=94; nonfinite=0.
- Categorical Ticket: common={'PC 17608': 5, '113503': 4, 'CA. 2343': 4, 'C.A. 31029': 3, 'PC 17483': 3, '347077': 3, 'SOTON/O.Q. 3101315': 3, '16966': 3, '220845': 3, '21228': 2}; singleton values=321; rare examples={'21228': 2, '24065': 2, '2662': 2, 'C.A. 2315': 2, 'W./C. 6607': 2, '13236': 2, '13695': 2, '19950': 2, 'F.C.C. 13534': 2, '2660': 2}.
- Numeric Fare: {'count': 417.0, 'mean': 35.627188489208635, 'std': 55.907576179973844, 'min': 0.0, '1%': 6.446828, '5%': 7.2292, '25%': 7.8958, '50%': 14.4542, '75%': 31.5, '95%': 151.55, '99%': 262.375, 'max': 512.3292}; skew=3.6872133081121405; 3-IQR extremes=31; nonfinite=0.
- Categorical Cabin: common={'B57 B59 B63 B66': 3, 'B45': 2, 'C78': 2, 'C31': 2, 'C23 C25 C27': 2, 'C101': 2, 'C55 C57': 2, 'C116': 2, 'C6': 2, 'F4': 2}; singleton values=62; rare examples={'B45': 2, 'C78': 2, 'C31': 2, 'C23 C25 C27': 2, 'C101': 2, 'C55 C57': 2, 'C116': 2, 'C6': 2, 'F4': 2, 'E34': 2}.
- Categorical Embarked: common={'S': 270, 'C': 102, 'Q': 46}; singleton values=0; rare examples={}.

## train.csv
Exact file rows: 891; columns: 12; analyzed rows: 891. COMPLETE TABLE.
Analyzed memory: 292,464 bytes.
Columns: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
Candidate IDs/keys (name heuristic): []
Exact duplicate rows in analyzed data: 0

| Column | Type | Missing count (%) | Unique (ratio) | Signals |
|---|---|---|---|---|
| PassengerId | int64 | 0 (0.00%) | 891 (1.000) | likely numeric/continuous; confirm semantics, near-unique; possible identifier |
| Survived | int64 | 0 (0.00%) | 2 (0.002) | likely categorical |
| Pclass | int64 | 0 (0.00%) | 3 (0.003) | likely categorical |
| Name | str | 0 (0.00%) | 891 (1.000) | likely categorical, high cardinality, near-unique; possible identifier |
| Sex | str | 0 (0.00%) | 2 (0.002) | likely categorical |
| Age | float64 | 177 (19.87%) | 88 (0.099) | likely numeric/continuous; confirm semantics |
| SibSp | int64 | 0 (0.00%) | 7 (0.008) | likely categorical |
| Parch | int64 | 0 (0.00%) | 7 (0.008) | likely categorical |
| Ticket | str | 0 (0.00%) | 681 (0.764) | likely categorical, high cardinality |
| Fare | float64 | 0 (0.00%) | 248 (0.278) | likely numeric/continuous; confirm semantics |
| Cabin | str | 687 (77.10%) | 147 (0.165) | likely categorical, high cardinality |
| Embarked | str | 2 (0.22%) | 3 (0.003) | likely categorical |

Top missingness masks (1=missing, column order above): {'000000000010': 529, '000000000000': 183, '000001000010': 158, '000001000000': 19, '000000000001': 2}
Duplicate candidate-feature rows excluding named ID candidates and any configured target: 0; confirm feature roles before interpretation.
- Numeric PassengerId: {'count': 891.0, 'mean': 446.0, 'std': 257.3538420152301, 'min': 1.0, '1%': 9.9, '5%': 45.5, '25%': 223.5, '50%': 446.0, '75%': 668.5, '95%': 846.5, '99%': 882.1, 'max': 891.0}; skew=0.0; 3-IQR extremes=0; nonfinite=0.
- Numeric Survived: {'count': 891.0, 'mean': 0.3838383838383838, 'std': 0.4865924542648575, 'min': 0.0, '1%': 0.0, '5%': 0.0, '25%': 0.0, '50%': 0.0, '75%': 1.0, '95%': 1.0, '99%': 1.0, 'max': 1.0}; skew=0.4785234382949897; 3-IQR extremes=0; nonfinite=0.
- Numeric Pclass: {'count': 891.0, 'mean': 2.308641975308642, 'std': 0.836071240977049, 'min': 1.0, '1%': 1.0, '5%': 1.0, '25%': 2.0, '50%': 3.0, '75%': 3.0, '95%': 3.0, '99%': 3.0, 'max': 3.0}; skew=-0.6305479068752845; 3-IQR extremes=0; nonfinite=0.
- Categorical Name: common={'Braund, Mr. Owen Harris': 1, 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)': 1, 'Heikkinen, Miss. Laina': 1, 'Futrelle, Mrs. Jacques Heath (Lily May Peel)': 1, 'Allen, Mr.; singleton values=891; rare examples={'Braund, Mr. Owen Harris': 1, 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)': 1, 'Heikkinen, Miss. Laina': 1, 'Futrelle, Mrs. Jacques Heath (Lily May Peel)': 1, 'Allen, Mr..
- Categorical Sex: common={'male': 577, 'female': 314}; singleton values=0; rare examples={}.
- Numeric Age: {'count': 714.0, 'mean': 29.69911764705882, 'std': 14.526497332334042, 'min': 0.42, '1%': 1.0, '5%': 4.0, '25%': 20.125, '50%': 28.0, '75%': 38.0, '95%': 56.0, '99%': 65.87, 'max': 80.0}; skew=0.38910778230082693; 3-IQR extremes=0; nonfinite=0.
- Numeric SibSp: {'count': 891.0, 'mean': 0.5230078563411896, 'std': 1.1027434322934317, 'min': 0.0, '1%': 0.0, '5%': 0.0, '25%': 0.0, '50%': 0.0, '75%': 1.0, '95%': 3.0, '99%': 5.0, 'max': 8.0}; skew=3.6953517271630565; 3-IQR extremes=12; nonfinite=0.
- Numeric Parch: {'count': 891.0, 'mean': 0.38159371492704824, 'std': 0.8060572211299483, 'min': 0.0, '1%': 0.0, '5%': 0.0, '25%': 0.0, '50%': 0.0, '75%': 0.0, '95%': 2.0, '99%': 4.0, 'max': 6.0}; skew=2.7491170471010933; 3-IQR extremes=213; nonfinite=0.
- Categorical Ticket: common={'347082': 7, '1601': 7, 'CA. 2343': 7, '3101295': 6, 'CA 2144': 6, '347088': 6, '382652': 5, 'S.O.C. 14879': 5, '349909': 4, '347077': 4}; singleton values=547; rare examples={'113803': 2, '237736': 2, 'PP 9549': 2, '239865': 2, 'PC 17569': 2, 'PC 17604': 2, '113789': 2, '345764': 2, '2651': 2, '11668': 2}.
- Numeric Fare: {'count': 891.0, 'mean': 32.204207968574636, 'std': 49.6934285971809, 'min': 0.0, '1%': 0.0, '5%': 7.225, '25%': 7.9104, '50%': 14.4542, '75%': 31.0, '95%': 112.07915, '99%': 249.00622000000035, 'max': 512.3292}; skew=4.787316519674893; 3-IQR extremes=53; nonfinite=0.
- Categorical Cabin: common={'G6': 4, 'C23 C25 C27': 4, 'B96 B98': 4, 'F33': 3, 'E101': 3, 'F2': 3, 'D': 3, 'C22 C26': 3, 'C123': 2, 'D33': 2}; singleton values=101; rare examples={'C123': 2, 'D33': 2, 'C52': 2, 'B28': 2, 'C83': 2, 'F G73': 2, 'D26': 2, 'B58 B60': 2, 'C2': 2, 'E33': 2}.
- Categorical Embarked: common={'S': 644, 'C': 168, 'Q': 77}; singleton values=0; rare examples={}.

## Roles, target, and train/test diagnostics
Sample submission schema: {'PassengerId': 'int64', 'Survived': 'int64'}; analyzed rows=418; consult file section for exact total.
Train-only columns / target candidates: ['Survived']
Test-only columns: []
Distinct matching train/test feature vectors in analyzed rows, excluding candidate IDs and target: 0. No labels transferred.
- PassengerId: missingness train=0.000%, test=0.000%.
  Numeric drift diagnostic: train quantiles={0.05: 45.5, 0.5: 446.0, 0.95: 846.5}; test quantiles={0.05: 912.85, 0.5: 1100.5, 0.95: 1288.15}.
- Pclass: missingness train=0.000%, test=0.000%.
  Numeric drift diagnostic: train quantiles={0.05: 1.0, 0.5: 3.0, 0.95: 3.0}; test quantiles={0.05: 1.0, 0.5: 3.0, 0.95: 3.0}.
- Name: missingness train=0.000%, test=0.000%.
  Category drift diagnostic: total variation=0.9978; train-only=889 ['Abbing, Mr. Anthony', 'Abbott, Mr. Rossmore Edward', 'Abbott, Mrs. Stanton (Rosa Hunt)', 'Abelson, Mr. Samuel', 'Abelson, Mrs. Samuel (Hannah Wizosky)', 'Adahl, Mr. Mauritz Nils ; test-only=416 ['Abbott, Master. Eugene Joseph', 'Abelseth, Miss. Karen Marie', 'Abelseth, Mr. Olaus Jorgensen', 'Abrahamsson, Mr. Abraham August Johannes', 'Abrahim, Mrs. Joseph (Sophie Halaut E.
- Sex: missingness train=0.000%, test=0.000%.
  Category drift diagnostic: total variation=0.0112; train-only=0 []; test-only=0 [].
- Age: missingness train=19.865%, test=20.574%.
  Numeric drift diagnostic: train quantiles={0.05: 4.0, 0.5: 28.0, 0.95: 56.0}; test quantiles={0.05: 8.0, 0.5: 27.0, 0.95: 57.0}.
- SibSp: missingness train=0.000%, test=0.000%.
  Numeric drift diagnostic: train quantiles={0.05: 0.0, 0.5: 0.0, 0.95: 3.0}; test quantiles={0.05: 0.0, 0.5: 0.0, 0.95: 2.0}.
- Parch: missingness train=0.000%, test=0.000%.
  Numeric drift diagnostic: train quantiles={0.05: 0.0, 0.5: 0.0, 0.95: 2.0}; test quantiles={0.05: 0.0, 0.5: 0.0, 0.95: 2.0}.
- Ticket: missingness train=0.000%, test=0.000%.
  Category drift diagnostic: total variation=0.7967; train-only=566 ['110152', '110413', '110465', '110564', '111240', '111320', '111361', '111369', '111426', '111427']; test-only=248 ['110469', '110489', '111163', '112051', '112377', '112378', '112901', '113038', '113044', '113054'].
- Fare: missingness train=0.000%, test=0.239%.
  Numeric drift diagnostic: train quantiles={0.05: 7.225, 0.5: 14.4542, 0.95: 112.07915}; test quantiles={0.05: 7.2292, 0.5: 14.4542, 0.95: 151.55}.
- Cabin: missingness train=77.104%, test=78.230%.
  Category drift diagnostic: total variation=0.7490; train-only=110 ['A10', 'A14', 'A16', 'A19', 'A20', 'A23', 'A24', 'A26', 'A31', 'A32']; test-only=39 ['A11', 'A18', 'A21', 'A29', 'A9', 'B10', 'B11', 'B24', 'B26', 'B36'].
- Embarked: missingness train=0.224%, test=0.000%.
  Category drift diagnostic: total variation=0.0785; train-only=0 []; test-only=0 [].

### Target analysis: Survived
Classes/unique values: 2; missing labels: 0.
Likely classification; numeric codes may still represent regression. Confirm from the problem statement.
Summary: {'count': 891.0, 'mean': 0.3838383838383838, 'std': 0.4865924542648575, 'min': 0.0, '25%': 0.0, '50%': 0.0, '75%': 1.0, 'max': 1.0}; distribution/common labels: {0: 549, 1: 342}; rare labels (<=2): 0.
Duplicate feature rows excluding target/candidate IDs: 0; feature groups with conflicting labels: 0.

## POTENTIAL LEAKAGE RISKS
Heuristics and drift signals are diagnostics, not proof. Review before using or removing features.
### CRITICAL
- No signal found by these checks; this does not establish absence of leakage.
### HIGH
- No signal found by these checks; this does not establish absence of leakage.
### MEDIUM
- No signal found by these checks; this does not establish absence of leakage.
### LOW
- No signal found by these checks; this does not establish absence of leakage.

Still requires human review: collection process, post-outcome features, time/groups/entities, duplicate policies, and fold-local preprocessing.
