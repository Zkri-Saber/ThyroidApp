# src/main.py

from thyroid_analysis.data_loader import load_excel_dataset
from thyroid_analysis.preprocessing import (
    remove_duplicates,
    convert_columns_to_numeric,
    enforce_column_types
)
from thyroid_analysis.eda import analyze_categorical_columns

def main():
    file_path = 'data/ExactRealDatasetLU.xlsx'
    
    # Load data
    df = load_excel_dataset(file_path)

    # Step 1: Remove duplicates
    df = remove_duplicates(df, verbose=True)

    # Step 2: Convert specific columns to numeric
    thyroid_columns = [
        'first TSH', 'last TSH', 'first T4', 'last T4',
        'first T3', 'first FT4', 'last FT4', 'first FT3', 'last FT3'
    ]
    df = convert_columns_to_numeric(df, thyroid_columns)

    # Step 3: Enforce correct data types
    df = enforce_column_types(df)

    # Step 4: Analyze categorical variables (optional in scripts)
    categorical_columns = ['Sex', 'Smoking', 'Marital status']
    analyze_categorical_columns(df, categorical_columns, show_plots=False)  # Set True if using in notebook

    # Final preview
    print("\nFinal preview of dataset:")
    print(df.head(10))


if __name__ == "__main__":
    main()
