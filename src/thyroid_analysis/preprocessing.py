# src/thyroid_analysis/preprocessing.py

import pandas as pd

def convert_thyroid_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert thyroid hormone columns to numeric types, coercing errors to NaN.
    """
    thyroid_columns = [
        'first TSH', 'last TSH', 'first T4', 'last T4',
        'first T3', 'first FT4', 'last FT4', 'first FT3', 'last FT3'
    ]
    df[thyroid_columns] = df[thyroid_columns].apply(pd.to_numeric, errors='coerce')
    return df

def enforce_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce appropriate data types for each column.
    """
    df['Age'] = df['Age'].astype(int)
    df['Sex'] = df['Sex'].astype('category')
    df['Smoking'] = df['Smoking'].astype('category')
    df['Marital status'] = df['Marital status'].astype('category')
    df['first TSH'] = df['first TSH'].astype(float)
    df['last TSH'] = df['last TSH'].astype(float)
    df['first T4'] = df['first T4'].astype(float)
    df['last T4'] = df['last T4'].astype(float)
    df['first T3'] = df['first T3'].astype(float)
    df['last T3'] = df['last T3'].astype(float)
    df['first FT4'] = df['first FT4'].astype(float)
    df['last FT4'] = df['last FT4'].astype(float)
    df['first FT3'] = df['first FT3'].astype(float)
    df['last FT3'] = df['last FT3'].astype(float)
    df['Dx'] = df['Dx'].astype('category')
    return df

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features: Sex, Smoking, Marital status
    """
    sex_mapping = {'Male': 0, 'Female': 1}
    smoking_mapping = {'No': 0, 'Passive': 1, 'Active': 2}
    marital_status_mapping = {'single': 0, 'married': 1}

    df['Sex'] = df['Sex'].map(sex_mapping)
    df['Smoking'] = df['Smoking'].map(smoking_mapping)
    df['Marital status'] = df['Marital status'].map(marital_status_mapping)

    return df

def map_diagnostic_group_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw Dx column to simplified diagnostic groups.
    """
    mapping = {
        'No Disease': 'No Disease',
        # Hyperthyroidism
        'Hyperthyroidisim': 'Hyperthyroidism',
        'Hyperthyroidisim, Multinodular Goiter (MNG)': 'Hyperthyroidism',
        'Graves Disease (GD), Hyperthyroidisim': 'Hyperthyroidism',
        'Hyperthyroidism, Multinodular Goiter (MNG)': 'Hyperthyroidism',
        'Hyperthyroidism': 'Hyperthyroidism',
        'Hyperthyroidisim, Thyroid Nodule': 'Hyperthyroidism',
        'hyperthyroid': 'Hyperthyroidism',
        'Graves Disease (GD), Hyperthyroidism': 'Hyperthyroidism',
        'Hyperthyroidism, Multinodular Goiter (MNG), Suspicious Thyroid Nodule': 'Hyperthyroidism',
        'Hyperthyroidism, Suspicious Thyroid Nodule': 'Hyperthyroidism',
        'hyper for 2 ys': 'Hyperthyroidism',
        'hyperthyroid for 15 month': 'Hyperthyroidism',
        'hyperthyroid for  3 ys': 'Hyperthyroidism',
        'hyperthyroid for 6 ys': 'Hyperthyroidism',
        # Euthyroid
        'euthyroid': 'Euthyroid',
        'Euthyroid, Thyroid Nodule': 'Euthyroid',
        'Euthyroid, Papillary Thyroid Carcinoma (PTC)': 'Euthyroid',
        'Euthyroid, Suspicious Thyroid Nodule': 'Euthyroid',
        'Euthyroid, Multinodular Goiter (MNG)': 'Euthyroid',
        'Euthyroid, Multinodular Goiter (MNG), Suspicious Thyroid Nodule': 'Euthyroid',
        'Euthyroid, Multinodular Goiter (MNG), RSE': 'Euthyroid',
        'Euthyroid, Hurthle Cell': 'Euthyroid',
        'Euthyroid, Hurthle Cell, Multinodular Goiter (MNG)': 'Euthyroid',
        'Euthyroid, Papillary Thyroid Carcinoma (PTC), Thyroid Nodule': 'Euthyroid',
        'Euthyroid, Papillary Thyroid Carcinoma (PTC), Positive Cervical LN': 'Euthyroid',
        'Euthyroid, Isthmus Nodule': 'Euthyroid',
        'Euthyroid, Goitor': 'Euthyroid',
        'Euthyroid, Nodular Colloid Goiter, Suspicious Thyroid Nodule': 'Euthyroid',
        'Euthyroid, Papillary Thyroid Microcarcinoma': 'Euthyroid',
        'Euthyroid, Medullary Thyroid Carcinoma': 'Euthyroid',
        'Euthyroid, Parathryoid Adenoma': 'Euthyroid',
        'Euthyroid, Follicular Thyroid Carcinoma (PTC)': 'Euthyroid',
        'Euthyroid, Papillary Thyroid Carcinoma (PTC), Suspicious Thyroid Nodule': 'Euthyroid',
        # Hypothyroidism
        'hypothyroid': 'Hypothyroidism',
        'Hypothyroidism, Suspicious Thyroid Nodule': 'Hypothyroidism',
        'Hypothyroidism, Papillary Thyroid Carcinoma (PTC)': 'Hypothyroidism',
        'Hypothyroidism, Thyroid Nodule': 'Hypothyroidism',
        'Hypothyroidism, Multinodular Goiter (MNG)': 'Hypothyroidism',
        'Hypothyroidism, Multinodular Goiter (MNG), Papillary Thyroid Carcinoma (PTC)': 'Hypothyroidism',
        'Hypothyroidism, RSE': 'Hypothyroidism',
        'Hypothyroidism, Papillary Thyroid Microcarcinoma': 'Hypothyroidism',
        'Hypothyroidism, Papillary Thyroid Carcinoma (PTC), Positive Cervical LN': 'Hypothyroidism',
        'Chronic Thyroiditis, Hypothyroidism': 'Hypothyroidism',
        'Hypoparathyroidism, Hypothyroidism': 'Hypothyroidism',
        'Hypothyroidism, Multinodular Goiter (MNG), Suspicious Thyroid Nodule': 'Hypothyroidism',
        'Hypothyroidism, Multinodular Goiter (MNG), Thyroid Nodule': 'Hypothyroidism',
        'Hypothyroidism, Papillary Thyroid Carcinoma (PTC), Thyroid Nodule': 'Hypothyroidism',
        'Hypoparathyroidism, Hypothyroidism, Papillary Thyroid Carcinoma (PTC)': 'Hypothyroidism',
    }

    df['Diagnostic Group'] = df['Dx'].map(mapping)
    print("Diagnostic group counts:")
    print(df['Diagnostic Group'].value_counts())

    return df

def encode_diagnostic_group_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Diagnostic Group column to numerical codes.
    """
    diagnostic_group_mapping = {
        'No Disease': 0,
        'Hyperthyroidism': 1,
        'Euthyroid': 2,
        'Hypothyroidism': 3
    }
    df['Diagnostic Group Code'] = df['Diagnostic Group'].map(diagnostic_group_mapping)
    print("\nDiagnostic Group unique values:")
    print(df['Diagnostic Group'].unique())
    return df

def drop_irrelevant_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Drop non-useful columns.
    """
    columns_to_drop = ['Info.ID', 'Name', 'Occupation', 'Indication']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    if verbose:
        print("\nRemaining columns after dropping irrelevant ones:")
        print(df.columns)

    return df
