# src/thyroid_analysis/preprocessing.py

import pandas as pd

def remove_duplicates(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    """
    original_count = len(df)
    df_cleaned = df.drop_duplicates()
    cleaned_count = len(df_cleaned)

    if verbose:
        print(f"Duplicates removed: {original_count - cleaned_count}")
        print(f"Original rows: {original_count}, Cleaned rows: {cleaned_count}")

    return df_cleaned


def convert_columns_to_numeric(df: pd.DataFrame, columns: list, verbose: bool = True) -> pd.DataFrame:
    """
    Convert specified columns to numeric, coercing errors to NaN.
    """
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')
    
    if verbose:
        print("\nData types after numeric conversion:")
        print(df[columns].dtypes)

    return df


def enforce_column_types(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Explicitly cast selected columns to appropriate data types.
    """
    type_map = {
        'Age': int,
        'Sex': 'category',
        'Smoking': 'category',
        'Marital status': 'category',
        'first TSH': float,
        'last TSH': float,
        'first T4': float,
        'last T4': float,
        'first T3': float,
        'last T3': float,
        'first FT4': float,
        'last FT4': float,
        'first FT3': float,
        'last FT3': float,
        'Dx': 'category'
    }

    for column, dtype in type_map.items():
        if column in df.columns:
            df[column] = df[column].astype(dtype)

    if verbose:
        print("\nData types after enforcing type casting:")
        print(df[list(type_map.keys())].dtypes)

    return df


def encode_categorical_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply predefined mappings to categorical columns: Sex, Smoking, Marital status.
    """
    sex_mapping = {'Male': 0, 'Female': 1}
    smoking_mapping = {'No': 0, 'Passive': 1, 'Active': 2}
    marital_status_mapping = {'single': 0, 'married': 1}

    df['Sex'] = df['Sex'].map(sex_mapping)
    df['Smoking'] = df['Smoking'].map(smoking_mapping)
    df['Marital status'] = df['Marital status'].map(marital_status_mapping)

    if verbose:
        print("\nEncoded values (Age, Sex, Smoking, Marital status):")
        print(df[['Age', 'Sex', 'Smoking', 'Marital status']].head())

        # Check for unmapped values
        if df['Sex'].isnull().any():
            print("\n⚠️ Unmapped 'Sex' values:")
            print(df[df['Sex'].isnull()])

        if df['Smoking'].isnull().any():
            print("\n⚠️ Unmapped 'Smoking' values:")
            print(df[df['Smoking'].isnull()])

        if df['Marital status'].isnull().any():
            print("\n⚠️ Unmapped 'Marital status' values:")
            print(df[df['Marital status'].isnull()])

    return df
