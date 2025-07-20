"""import pandas as pd
import numpy as np
from fancyimpute import KNN, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from .config import NUMERIC_COLUMNS, KNN_K

def convert_to_numeric(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    cols = cols or NUMERIC_COLUMNS
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    return df

def impute_knn(df: pd.DataFrame, k: int = KNN_K) -> pd.DataFrame:
    # Only impute numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    imputer = KNN(k=k)
    imputed = imputer.fit_transform(df[num_cols])
    df[num_cols] = imputed
    return df

def impute_mice(df: pd.DataFrame) -> pd.DataFrame:
    # Only impute numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    imputer = IterativeImputer(random_state=42)
    imputed = imputer.fit_transform(df[num_cols])
    df[num_cols] = imputed
    return df

def detect_and_remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Use only numeric data to detect outliers
    numeric = df.select_dtypes(include=[np.number])
    iso = IsolationForest(contamination=0.01, random_state=42)
    mask = iso.fit_predict(numeric)
    return df.loc[mask == 1].reset_index(drop=True)

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = scaler.fit_transform(df[numeric])
    return df
"""

import pandas as pd
import numpy as np
from fancyimpute import KNN, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from .config import NUMERIC_COLUMNS, KNN_K

def convert_to_numeric(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    cols = cols or NUMERIC_COLUMNS
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    return df

def impute_knn(df: pd.DataFrame, k: int = KNN_K) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    imputer = KNN(k=k)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df

def impute_mice(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    imputer = IterativeImputer(random_state=42)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df

def detect_and_remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    iso = IsolationForest(contamination=0.01, random_state=42)
    mask = iso.fit_predict(numeric)
    return df.loc[mask == 1].reset_index(drop=True)

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df