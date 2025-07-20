import pandas as pd
from .data_loader import load_data
from .preprocessing import (
    convert_to_numeric, impute_knn, impute_mice,
    detect_and_remove_outliers, standardize
)
from .feature_selection import select_by_rfe, select_by_pca
from .modeling import train_model, evaluate_model

def run_pipeline() -> pd.DataFrame:
    # 1. Load & initial clean
    df = load_data()
    df = convert_to_numeric(df)
    df = df.dropna(subset=["Diagnostic Group Code"])

    # 2. Impute & clean
    df = impute_knn(df)
    df = impute_mice(df)
    df = detect_and_remove_outliers(df)
    df = standardize(df)

    # 3. Split features/target
    X = df.drop(columns=["Diagnostic Group Code"])
    y = df["Diagnostic Group Code"]

    # 4. Feature selection variants
    X_rfe = select_by_rfe(X, y)
    X_pca = select_by_pca(X)

    # 5. Train & evaluate
    results = []
    for method, X_sel in [("RFE", X_rfe), ("PCA", X_pca)]:
        model = train_model(X_sel, y, model_name="random_forest")
        metrics = evaluate_model(model, X_sel, y)
        metrics["feature_method"] = method
        results.append(metrics)

    return pd.DataFrame(results)
