import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from .config import RANDOM_FOREST_PARAMS, SMOTE_SAMPLING_STRATEGY

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "random_forest"
):
    """
    Train with SMOTE and return fitted estimator.
    """
    if model_name == "random_forest":
        model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    elif model_name == "svm":
        model = SVC(probability=True, random_state=42)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    smote = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    model.fit(X_res, y_res)
    return model

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.Series:
    """
    Return accuracy, precision, recall, f1, and confusion matrix.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return pd.Series(metrics)
