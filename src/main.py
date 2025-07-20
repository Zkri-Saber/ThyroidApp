# src/main.py

from thyroid_analysis.data_loader import load_excel_dataset
from thyroid_analysis.preprocessing import (
    remove_duplicates,
    convert_columns_to_numeric,
    enforce_column_types,
    encode_categorical_columns,
    map_diagnostic_groups
)
from thyroid_analysis.eda import analyze_categorical_columns

def main():
    file_path = 'data/ExactRealDatasetLU.xlsx'
    
    # Step 0: Load the dataset
    df = load_excel_dataset(file_path)

    # Step 1: Remove duplicate rows
    df = remove_duplicates(df, verbose=True)

    # Step 2: Convert thyroid-related columns to numeric
    thyroid_columns = [
        'first TSH', 'last TSH', 'first T4', 'last T4',
        'first T3', 'first FT4', 'last FT4', 'first FT3', 'last FT3'
    ]
    df = convert_columns_to_numeric(df, thyroid_columns)

    # Step 3: Enforce correct data types
    df = enforce_column_types(df)

    # Step 4: Encode Sex, Smoking, Marital status
    df = encode_categorical_columns(df)

    # Step 5: Map diagnostic group from 'Dx'
    df = map_diagnostic_groups(df)

    # Step 6: Analyze and save bar charts for categorical columns
    categorical_columns = ['Sex', 'Smoking', 'Marital status']
    analyze_categorical_columns(
        df,
        columns=categorical_columns,
        show_plots=False,
        save_dir="outputs/eda"
    )

    # Final preview
    print("\nPreview of final dataset:")
    print(df.head(10))


if __name__ == "__main__":
    main()
