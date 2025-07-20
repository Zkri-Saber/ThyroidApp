import pandas as pd
from .data_loader import load_data
from .preprocessing import (
    convert_to_numeric,
    impute_knn,
    impute_mice,
    detect_and_remove_outliers,
    standardize
)
from .feature_selection import select_by_rfe, select_by_pca
from .modeling import train_model, evaluate_model
from .config import TARGET_COLUMN, NUMERIC_COLUMNS

def run_pipeline() -> pd.DataFrame:
    # 1. Load & initial clean
    df = load_data()
    df = convert_to_numeric(df)

    # Sanity-check that the expected target column exists
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found in data.\n"
            f"Available columns: {df.columns.tolist()}"
        )
    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET_COLUMN])

    # 2. Impute & clean (numeric only)
    df = impute_knn(df)
    df = impute_mice(df)
    df = detect_and_remove_outliers(df)
    df = standardize(df)

    # 3. Split features / target (numeric features only)
    X = df[NUMERIC_COLUMNS]
    y = df[TARGET_COLUMN]

    # 4. Feature-selection variants
    X_rfe = select_by_rfe(X, y)
    X_pca = select_by_pca(X)

    # 5. Train & evaluate
    results = []
    for method, X_sel in [("RFE", X_rfe), ("PCA", X_pca)]:
        model   = train_model(X_sel, y, model_name="random_forest")
        metrics = evaluate_model(model, X_sel, y)
        metrics["feature_method"] = method
        results.append(metrics)

    return pd.DataFrame(results)
