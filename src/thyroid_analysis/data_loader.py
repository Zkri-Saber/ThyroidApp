import pandas as pd
from .config import EXACT_REAL_DATASET, SHEET_NAME

def load_data(path: str = None) -> pd.DataFrame:
    """
    Load the thyroid dataset from Excel, drop duplicates, reset index.
    """
    file_path = path or EXACT_REAL_DATASET
    df = pd.read_excel(file_path, sheet_name=SHEET_NAME)
    df = df.drop_duplicates().reset_index(drop=True)
    return df
