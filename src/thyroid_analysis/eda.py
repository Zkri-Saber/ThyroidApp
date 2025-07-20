# src/thyroid_analysis/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_categorical_columns(df: pd.DataFrame, columns: list, show_plots: bool = True):
    """
    Print frequency tables and plot bar charts for each categorical column.

    Parameters:
    - df (pd.DataFrame): The dataset
    - columns (list): List of categorical column names
    - show_plots (bool): If True, show bar charts
    """
    for column in columns:
        print(f"\nFrequency Table for '{column}':")
        print(df[column].value_counts())

    if show_plots:
        for column in columns:
            plt.figure(figsize=(16, 8))
            sns.countplot(x=df[column])
            plt.title(f'Bar Chart of {column}')
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
