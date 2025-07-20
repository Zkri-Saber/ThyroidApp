# src/main.py

import pandas as pd
from thyroid_analysis.data_loader import load_excel_dataset
from thyroid_analysis.preprocessing import (
    convert_thyroid_columns_to_numeric,
    enforce_column_types,
    encode_categorical_columns,
    map_diagnostic_group_column,
    encode_diagnostic_group_column,
    drop_irrelevant_columns
)
from thyroid_analysis.eda import (
    analyze_categorical_columns,
    analyze_numerical_columns
)

def main():
    # Step 1: Load Excel dataset
    file_path = 'data/ExactRealDatasetLU.xlsx'
    df = load_excel_dataset(file_path, sheet_name='Sheet1')

    # Step 2: Convert hormone columns to numeric
    df = convert_thyroid_columns_to_numeric(df)

    # Step 3: Enforce correct data types
    df = enforce_column_types(df)

    # Step 4: Encode Sex, Smoking, Marital status
    df = encode_categorical_columns(df)

    # Step 5: Map Dx column into 4 groups
    df = map_diagnostic_group_column(df)

    # Step 6: Encode group as 0/1/2/3
    df = encode_diagnostic_group_column(df)

    # Step 7: Drop irrelevant columns
    df = drop_irrelevant_columns(df)

    # Step 8: Categorical EDA
    categorical_columns = ['Sex', 'Smoking', 'Marital status', 'Diagnostic Group']
    analyze_categorical_columns(df, columns=categorical_columns, show_plots=False, save_dir="outputs/eda/categorical")

    # Step 9: Numerical EDA
    numerical_columns = [
        'Age', 'first TSH', 'last TSH', 'first T3', 'last T3',
        'first T4', 'last T4', 'Smoking', 'Marital status',
        'first FT3', 'last FT3'
    ]
    analyze_numerical_columns(df, columns=numerical_columns, show_plots=False, save_dir="outputs/eda/numerical")

    # Step 10: Final preview
    print("\nFinal Data Sample:")
    print(df.head())

if __name__ == "__main__":
    main()
