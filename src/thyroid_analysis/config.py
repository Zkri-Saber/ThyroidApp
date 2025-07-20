import os
from pathlib import Path

# Base directories (project root → data, outputs)
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# Dataset settings
EXACT_REAL_DATASET = DATA_DIR / "ExactRealDatasetLU.xlsx"
SHEET_NAME         = "Sheet1"

# Target column name in your Excel sheet
TARGET_COLUMN      = "Dx"   # updated to match your actual header

# Columns
NUMERIC_COLUMNS    = [
    "first TSH", "last TSH",
    "first T4",  "last T4",
    "first T3",  "last T3",
    "first FT4", "last FT4",
    "first FT3", "last FT3"
]
CATEGORICAL_COLUMNS = [
    "Info.ID", "Name", "Age", "Sex", "Occupation",
    "Smoking", "Marital status", "Indication"
]

# Imputation
KNN_K = 5

# Feature selection
RFE_FEATURE_COUNT = 10
PCA_COMPONENTS    = 5

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "random_state": 42
}

# SMOTE settings
SMOTE_SAMPLING_STRATEGY = "auto"
