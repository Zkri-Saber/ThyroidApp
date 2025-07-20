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
    imputer = KNN(k=k)
    df[df.columns] = imputer.fit_transform(df)
    return df

def impute_mice(df: pd.DataFrame) -> pd.DataFrame:
    imputer = IterativeImputer(random_state=42)
    df[df.columns] = imputer.fit_transform(df)
    return df

def detect_and_remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    iso = IsolationForest(contamination=0.01, random_state=42)
    numeric = df.select_dtypes(include=[np.number])
    mask = iso.fit_predict(numeric)
    return df[mask == 1].reset_index(drop=True)

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    numeric = df.select_dtypes(include=[np.number])
    df[numeric.columns] = scaler.fit_transform(numeric)
    return df
