import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# Dataset settings
EXACT_REAL_DATASET = DATA_DIR / "ExactRealDatasetLU.xlsx"
SHEET_NAME = "Sheet1"

# Columns
NUMERIC_COLUMNS = [
    "first TSH", "last TSH",
    "first T4", "last T4",
    "first T3", "first FT4",
    "last FT4", "first FT3",
    "last FT3"
]
CATEGORICAL_COLUMNS = []  # add names here if you have any

# Imputation
KNN_K = 5

# Feature selection
RFE_FEATURE_COUNT = 10
PCA_COMPONENTS = 5

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "random_state": 42
}

# SMOTE
SMOTE_SAMPLING_STRATEGY = "auto"
