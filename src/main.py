# src/main.py

import os
import pandas as pd
import altair as alt
import datetime
import numpy as np
from scipy.stats import entropy

from thyroid_analysis.data_loader import load_excel_dataset
from thyroid_analysis.preprocessing import (
    convert_thyroid_columns_to_numeric,
    enforce_column_types,
    encode_categorical_columns,
    map_diagnostic_group_column,
    encode_diagnostic_group_column,
    drop_irrelevant_columns,
    impute_missing_values_knn,
    impute_missing_values_mice,
)
from thyroid_analysis.eda import (
    analyze_categorical_columns,
    analyze_numerical_columns,
    visualize_missing_data
)
from thyroid_analysis.KL_divergence import compute_kl_for_all_features, plot_kl_divergence

def main():
    file_path = 'data/ExactRealDatasetLU.xlsx'
    df = load_excel_dataset(file_path)

    # =========== Data Cleaning ===========
    df = convert_thyroid_columns_to_numeric(df)
    df = enforce_column_types(df)
    df = encode_categorical_columns(df)
    df = map_diagnostic_group_column(df)
    df = encode_diagnostic_group_column(df)
    df = drop_irrelevant_columns(df)

    # =========== EDA ===========
    analyze_categorical_columns(df, ['Sex', 'Smoking', 'Marital status'], show_plots=False, save_dir="outputs/eda")

    numerical_columns = ['Age', 'first TSH', 'last TSH', 'first T3', 'last T3', 'first T4', 'last T4',
                         'Smoking', 'Marital status', 'first FT3', 'last FT3']
    analyze_numerical_columns(df, numerical_columns, show_plots=False, save_dir="outputs/eda")

    # =========== Save Original and Visualize Missing ===========
    df_original = df.copy()
    visualize_missing_data(df_original, stage="before", save_dir="outputs/eda/imputed")

    # =========== Impute ===========
    df_knn_imputed = impute_missing_values_knn(df_original)
    visualize_missing_data(df_knn_imputed, stage="after_knn", save_dir="outputs/eda/imputed")

    df_mice_imputed = impute_missing_values_mice(df_original)
    visualize_missing_data(df_mice_imputed, stage="after_mice", save_dir="outputs/eda/imputed")

    # =========== Compare Multiple Columns Across Original, KNN, MICE ===========
    columns_to_compare = ['Age', 'first TSH', 'last TSH', 'first T3', 'last T3',
                          'first T4', 'last T4', 'Smoking', 'Marital status',
                          'first FT3', 'last FT3']

    # Use dictionary to store all columns at once
    comparison_data = {}
    for col in columns_to_compare:
        comparison_data[f'{col} Original'] = df_original[col]
        comparison_data[f'{col} KNN Imputed'] = df_knn_imputed[col]
        comparison_data[f'{col} MICE Imputed'] = df_mice_imputed[col]

    comparison_df = pd.DataFrame(comparison_data)
    comparison_dir = "outputs/eda/imputed"
    os.makedirs(comparison_dir, exist_ok=True)

    full_comparison_path = os.path.join(comparison_dir, "multi_column_imputation_comparison.csv")
    comparison_df.to_csv(full_comparison_path, index=False)
    print(f"✅ Saved full multi-column comparison to: {full_comparison_path}")

    nan_rows = comparison_df[[f'{col} Original' for col in columns_to_compare]].isna().any(axis=1)
    nan_comparison_df = comparison_df[nan_rows]

    # Save only the NaN-related rows with a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nan_filename = f"multi_column_nan_rows_comparison_{timestamp}.csv"
    nan_comparison_path = os.path.join(comparison_dir, nan_filename)
    nan_comparison_df.to_csv(nan_comparison_path, index=False)
    print(f"✅ Saved NaN-filtered comparison to: {nan_comparison_path}")

    print("\n🔍 Preview of rows with missing original values:")
    print(nan_comparison_df.head(10))
    print("🔍 Columns in comparison_df:", comparison_df.columns.tolist())
    print("🧪 Sample:\n", comparison_df.head())

    # =========== ✅ Save Imputed Datasets ===========
    output_dir = "outputs/datasets"
    os.makedirs(output_dir, exist_ok=True)

    knn_path = os.path.join(output_dir, "real_dataset_knn_imputed.csv")
    mice_path = os.path.join(output_dir, "real_dataset_mice_imputed.csv")
    print("✅ KNN Imputed Shape:", df_knn_imputed.shape)
    print("✅ MICE Imputed Shape:", df_mice_imputed.shape)

    df_knn_imputed.to_csv(knn_path, index=False)
    print(f"✅ KNN-imputed dataset saved to: {knn_path}")

    df_mice_imputed.to_csv(mice_path, index=False)
    print(f"✅ MICE-imputed dataset saved to: {mice_path}")

    print("✅ Saving to directory:", os.path.abspath(output_dir))
    print("✅ File exists after save (KNN)?", os.path.exists(knn_path))
    print("✅ File exists after save (MICE)?", os.path.exists(mice_path))

    # =========== 📊 KL Divergence Evaluation ===========
    kl_df = compute_kl_for_all_features(df_original, df_knn_imputed, df_mice_imputed, columns_to_compare)
    plot_kl_divergence(kl_df)

    # Save KL results to timestamped CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    kl_filename = f"kl_divergence_comparison_{timestamp}.csv"
    kl_output_path = os.path.join(comparison_dir, kl_filename)
    kl_df.to_csv(kl_output_path, index=False)
    print(f"✅ KL divergence results saved to: {kl_output_path}")

if __name__ == "__main__":
    main()