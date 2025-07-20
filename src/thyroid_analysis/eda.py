# src/thyroid_analysis/eda.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_categorical_columns(df: pd.DataFrame, columns: list, show_plots: bool = True, save_dir: str = "outputs/eda"):
    """
    Print frequency tables and plot bar charts for each categorical column.
    Saves plots to specified directory.

    Parameters:
    - df (pd.DataFrame): The dataset
    - columns (list): List of categorical column names
    - show_plots (bool): If True, displays plots
    - save_dir (str): Directory path where plots will be saved
    """
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    for column in columns:
        print(f"\nFrequency Table for '{column}':")
        print(df[column].value_counts())

        plt.figure(figsize=(12, 6))
        sns.countplot(x=df[column])
        plt.title(f'Bar Chart of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = os.path.join(save_dir, f"{column.replace(' ', '_')}_bar_chart.png")
        plt.savefig(filename)
        print(f"Saved chart: {filename}")

        if show_plots:
            plt.show()
        else:
            plt.close()
