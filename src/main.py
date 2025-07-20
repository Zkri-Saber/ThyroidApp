# src/main.py

from thyroid_analysis.data_loader import load_excel_dataset
from thyroid_analysis.preprocessing import (
    remove_duplicates,
    convert_columns_to_numeric,
    enforce_column_types
)
from thyroid_analysis.eda import analyze_categorical_columns

def main():
    # Step 0: Load the dataset
    file_path = 'data/ExactRealDatasetLU.xlsx'
    df = load_excel_dataset(file_path)

    # Step 1: Remove duplicate rows
    df = remove_duplicates(df, verbose=True)

    # Step 2: Convert relevant thyroid columns to numeric (coerce errors)
    thyroid_columns = [
        'first TSH', 'last TSH', 'first T4', 'last T4',
        'first T3', 'first FT4', 'last FT4', 'first FT3', 'last FT3'
    ]
    df = convert_columns_to_numeric(df, thyroid_columns)

    # Step 3: Enforce correct data types
    df = enforce_column_types(df)

    # Step 4: Analyze categorical variables
    categorical_columns = ['Sex', 'Smoking', 'Marital status']
    analyze_categorical_columns(
        df,
        columns=categorical_columns,
        show_plots=False,  # Set to True if you want to view them interactively
        save_dir="outputs/eda"  # Save visualizations here
    )

    # Step 5: Final data preview
    print("\nPreview of final cleaned and typed dataset:")
    print(df.head(10))


if __name__ == "__main__":
    main()
