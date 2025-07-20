# src/main.py

from thyroid_analysis.data_loader import load_excel_dataset
from thyroid_analysis.preprocessing import (
    remove_duplicates,
    convert_columns_to_numeric,
    enforce_column_types,
    encode_categorical_columns
)
from thyroid_analysis.eda import analyze_categorical_columns

def main():
    file_path = 'data/ExactRealDatasetLU.xlsx'
    
    # Step 0: Load data
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

    # Step 4: Encode categorical columns
    df = encode_categorical_columns(df, verbose=True)

    # Step 5: Analyze and save categorical column bar charts
    categorical_columns = ['Sex', 'Smoking', 'Marital status']
    analyze_categorical_columns(
        df,
        columns=categorical_columns,
        show_plots=False,
        save_dir="outputs/eda"
    )

    # Step 6: Final data preview
    print("\nPreview of final cleaned and encoded dataset:")
    print(df.head(10))


if __name__ == "__main__":
    main()
