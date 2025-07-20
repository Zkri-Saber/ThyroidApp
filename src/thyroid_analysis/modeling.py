"""import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from .config import RANDOM_FOREST_PARAMS, SMOTE_SAMPLING_STRATEGY, KNN_K
from collections import Counter


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "random_forest"
):
    """
    Train a classifier, applying SMOTE only if every class has at least 2 samples.
    """
    # Choose model
    if model_name == "random_forest":
        model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    elif model_name == "svm":
        model = SVC(probability=True, random_state=42)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Check class counts
    class_counts = Counter(y)
    min_count = min(class_counts.values())

    if min_count < 2:
        # Too few samples for SMOTE; train on original data
        model.fit(X, y)
        return model

    # Determine k_neighbors (must be < min_count)
    k_neighbors = min(KNN_K, min_count - 1)

    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        k_neighbors=k_neighbors,
        random_state=42
    )
    X_res, y_res = smote.fit_resample(X, y)
    model.fit(X_res, y_res)
    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.Series:
    """
    Return accuracy, precision, recall, f1 (macro & weighted),
    and confusion matrix for multiclass.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        # macro averages across classes equally
        "precision_macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
        # weighted accounts for class support
        "precision_weighted": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return pd.Series(metrics)
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from .config import RANDOM_FOREST_PARAMS, SMOTE_SAMPLING_STRATEGY, KNN_K
from collections import Counter


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "random_forest"
):
    if model_name == "random_forest":
        model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    elif model_name == "svm":
        model = SVC(probability=True, random_state=42)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    class_counts = Counter(y)
    min_count = min(class_counts.values())
    if min_count < 2:
        model.fit(X, y)
        return model
    k_neighbors = min(KNN_K, min_count - 1)
    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        k_neighbors=k_neighbors,
        random_state=42
    )
    X_res, y_res = smote.fit_resample(X, y)
    model.fit(X_res, y_res)
    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.Series:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return pd.Series(metrics)