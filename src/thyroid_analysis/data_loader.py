import pandas as pd

def load_excel_dataset(file_path: str, sheet_name: str = 'Sheet1') -> pd.DataFrame:
    """
    Load a specified sheet from an Excel file.
    """
    sheets = pd.read_excel(file_path, sheet_name=None)
    if sheet_name not in sheets:
        raise ValueError(f"Sheet '{sheet_name}' not found in {file_path}. Available sheets: {list(sheets.keys())}")
    return sheets[sheet_name]
