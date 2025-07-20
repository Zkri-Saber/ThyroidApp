
# src/thyroid_analysis/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

def convert_thyroid_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    thyroid_columns = [
        'first TSH', 'last TSH', 'first T4', 'last T4',
        'first T3', 'first FT4', 'last FT4', 'first FT3', 'last FT3'
    ]
    df[thyroid_columns] = df[thyroid_columns].apply(pd.to_numeric, errors='coerce')
    return df

def enforce_column_types(df: pd.DataFrame) -> pd.DataFrame:
    df['Age'] = df['Age'].astype(int)
    df['Sex'] = df['Sex'].astype('category')
    df['Smoking'] = df['Smoking'].astype('category')
    df['Marital status'] = df['Marital status'].astype('category')
    df['Dx'] = df['Dx'].astype('category')
    for col in ['first TSH', 'last TSH', 'first T4', 'last T4',
                'first T3', 'last T3', 'first FT4', 'last FT4',
                'first FT3', 'last FT3']:
        df[col] = df[col].astype(float)
    return df

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    sex_mapping = {'Male': 0, 'Female': 1}
    smoking_mapping = {'No': 0, 'Passive': 1, 'Active': 2}
    marital_status_mapping = {'single': 0, 'married': 1}

    df['Sex'] = df['Sex'].map(sex_mapping)
    df['Smoking'] = df['Smoking'].map(smoking_mapping)
    df['Marital status'] = df['Marital status'].map(marital_status_mapping)

    return df

def map_diagnostic_group_column(df: pd.DataFrame) -> pd.DataFrame:
    from .diagnostic_mapping import diagnostic_mapping
    df['Diagnostic Group'] = df['Dx'].map(diagnostic_mapping)
    print("Diagnostic group counts:")
    print(df['Diagnostic Group'].value_counts())
    return df

def encode_diagnostic_group_column(df: pd.DataFrame) -> pd.DataFrame:
    from .diagnostic_mapping import diagnostic_group_mapping
    df.loc[:, 'Diagnostic Group Code'] = df['Diagnostic Group'].map(diagnostic_group_mapping)
    print("\nDiagnostic Group unique values:")
    print(df['Diagnostic Group'].unique())
    return df

def drop_irrelevant_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    columns_to_drop = ['Info.ID', 'Name', 'Occupation', 'Indication']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    if verbose:
        print("\nRemaining columns after dropping irrelevant ones:")
        print(df.columns)
    return df

def impute_missing_values_knn(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_copy = df.copy()
    df_copy[numeric_columns] = knn_imputer.fit_transform(df[numeric_columns])

    print("Missing values after KNN Imputation:")
    print(df_copy.isnull().sum())
    return df_copy

def impute_missing_values_mice(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    mice_imputer = IterativeImputer()
    df_copy = df.copy()
    df_copy[numeric_columns] = mice_imputer.fit_transform(df[numeric_columns])

    print("Missing values after MICE Imputation:")
    print(df_copy.isnull().sum())
    return df_copy

    
