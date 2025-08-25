# Data Card: Diabetes Prediction Dataset

## Source
- Local file: `./diabetes_prediction_dataset.csv`
- Provenance: Provided with project; public origin not specified
- License: TBD (document when known)

## Description
Tabular dataset for diabetes risk classification. Each row represents one individual with demographics, risk factors, and lab surrogates.

## Columns
- `gender` (categorical): Male, Female (others may be rare/absent)
- `age` (number): age in years
- `hypertension` (binary number): 0=no, 1=yes
- `heart_disease` (binary number): 0=no, 1=yes
- `smoking_history` (categorical): e.g., never, current, former, not current, ever, No Info
- `bmi` (number): body mass index
- `HbA1c_level` (number): percent
- `blood_glucose_level` (number): mg/dL
- `diabetes` (binary number): target label (0=no, 1=yes)

## Target
- Column: `diabetes`

## Preprocessing Plan
- Missing values: median imputation for numeric; most-frequent for categorical
- Scaling: standardize only for SVM/MLP
- Encoding: one-hot encode categoricals (`gender`, `smoking_history`)

## Split Configuration
- Train 70%, Validation 15%, Test 15%
- Stratified by target `diabetes`, seed=42

## Caveats & Biases
- Potential class imbalance; tune threshold for high recall while keeping specificity reasonable
- Self-reported fields (e.g., smoking) may be noisy
- Not clinical-grade; demo-quality only; not a diagnostic device

## Privacy
- No PHI persistence; logs must exclude sensitive data
- Secrets stored server-side via environment variables
