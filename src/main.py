# src/main.py

from thyroid_analysis.data_loader import load_excel_dataset
from thyroid_analysis.preprocessing import (
    remove_duplicates,
    convert_columns_to_numeric,
    enforce_column_types,
    encode_categorical_columns,
    map_diagnostic_groups,
    encode_diagnostic_group
)
from thyroid_analysis.eda import analyze_categorical_columns

def main():
    file_path = 'data/ExactRealDatasetLU.xlsx'
    
    # Step 0: Load dataset
    df = load_excel_dataset(file_path)

    # Step 1: Remove duplicate rows
    df = remove_duplicates(df, verbose=True)

    # Step 2: Convert thyroid-related columns to numeric
    thyroid_columns = [
        'first TSH', 'last TSH', 'first T4', 'last T4',
        'first T3', 'first FT4', 'last FT4', 'first FT3', 'last FT3'
    ]
    df = convert_columns_to_numeric(df, thyroid_columns)

    # Step 3: Enforce proper data types
    df = enforce_column_types(df)

    # Step 4: Encode Sex, Smoking, Marital status
    df = encode_categorical_columns(df)

    # Step 5: Map 'Dx' to simplified diagnostic group
    df = map_diagnostic_groups(df)

    # Step 6: Encode diagnostic group as numeric label
    df = encode_diagnostic_group(df)

    # Step 7: Save bar plots of categorical features
    categorical_columns = ['Sex', 'Smoking', 'Marital status']
    analyze_categorical_columns(
        df,
        columns=categorical_columns,
        show_plots=False,
        save_dir="outputs/eda"
    )

    # Step 8: Final preview
    print("\nFinal preview of processed dataset:")
    print(df.head(10))


if __name__ == "__main__":
    main()
