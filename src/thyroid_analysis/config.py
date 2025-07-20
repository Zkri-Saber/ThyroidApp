# src/thyroid_analysis/config.py

# ========== Core Python ========== #
import os
import warnings
from collections import Counter

# ========== Data & Analysis ========== #
import numpy as np
import pandas as pd

# ========== Visualization ========== #
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from matplotlib_venn import venn3
import altair as alt

# ========== Machine Learning Models ========== #
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# ========== Preprocessing & Feature Engineering ========== #
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

# ========== Imbalanced Data Handling ========== #
from imblearn.over_sampling import SMOTE

# ========== Model Evaluation & Metrics ========== #
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_predict, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix
)

# ========== Statistical Imputation ========== #
from statsmodels.imputation import mice

# ========== Google Colab ========== #
try:
    from google.colab import drive
except ImportError:
    drive = None  # Optional: If not running in Colab

# ========== Suppress Warnings ========== #
warnings.filterwarnings("ignore")
