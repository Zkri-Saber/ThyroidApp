# src/main.py

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
from thyroid_analysis.pipeline import df
from thyroid_analysis.eda import (
    analyze_categorical_columns,
    analyze_numerical_columns
)

import altair as alt


def main():
    file_path = 'data/ExactRealDatasetLU.xlsx'

    # Step 0: Load dataset
    df = load_excel_dataset(file_path)

    # Step 1: Convert thyroid-related columns to numeric
    df = convert_thyroid_columns_to_numeric(df)

    # Step 2: Enforce proper data types
    df = enforce_column_types(df)

    # Step 3: Encode Sex, Smoking, Marital status
    df = encode_categorical_columns(df)

    # Step 4: Map 'Dx' to simplified diagnostic group
    df = map_diagnostic_group_column(df)

    # Step 5: Encode diagnostic group as numeric label
    df = encode_diagnostic_group_column(df)

    # Step 6: Drop irrelevant columns
    df = drop_irrelevant_columns(df)

    # Step 7: Save bar plots of categorical features
    categorical_columns = ['Sex', 'Smoking', 'Marital status']
    analyze_categorical_columns(
        df,
        columns=categorical_columns,
        show_plots=False,
        save_dir="outputs/eda"
    )

    # Step 8: EDA for numerical columns
    numerical_columns = ['Age', 'first TSH', 'last TSH', 'first T3', 'last T3', 'first T4', 'last T4',
                         'Smoking', 'Marital status', 'first FT3', 'last FT3']
    analyze_numerical_columns(
        df,
        columns=numerical_columns,
        show_plots=False,
        save_dir="outputs/eda"
    )

    # Step 9: Apply diagnostic group mapping using .loc
    diagnostic_group_mapping = {
        'No Disease': 0,
        'Hyperthyroidism': 1,
        'Euthyroid': 2,
        'Hypothyroidism': 3
    }
    df.loc[:, 'Diagnostic Group Code'] = df['Diagnostic Group'].map(diagnostic_group_mapping)

    print(df.head())

    # Step 10: Categorical variable frequency tables and charts
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns

    for col in categorical_cols:
        freq_table = df[col].value_counts().reset_index()
        freq_table.columns = [col, 'Frequency']
        print(f"\nFrequency Table for {col}:\n", freq_table)

    for col in categorical_cols:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(col + ':N', sort='-y'),
            y='count()',
            tooltip=[col, 'count()']
        ).properties(
            title=f'Bar Chart of {col}'
        ).interactive()

        chart.save(f'{col}_barchart.json')

    # Step 11: Impute missing values using KNN and MICE
    print("Missing values before imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0]) 
    if df.isnull().sum().any():
        print("Imputing missing values...")
    # Impute using KNN

    df = impute_missing_values_knn(df)
    print("After KNN Imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    df = impute_missing_values_mice(df)
    print("After MICE Imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])


if __name__ == "__main__":
    main()
