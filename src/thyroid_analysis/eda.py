# src/thyroid_analysis/eda.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

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


def analyze_numerical_columns(df: pd.DataFrame, columns: list, show_plots: bool = True, save_dir: str = "outputs/eda"):
    """
    Generate histograms and boxplots for a list of numerical columns.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - columns (list): List of numerical column names to plot.
    - show_plots (bool): Whether to display plots.
    - save_dir (str): Directory path where plots will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    for column in columns:
        plt.figure(figsize=(12, 5))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[column], kde=True)
        plt.title(f'Histogram of {column}')

        # Box Plot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[column])
        plt.title(f'Box Plot of {column}')

        plt.tight_layout()
        filename = os.path.join(save_dir, f"{column.replace(' ', '_')}_eda_plot.png")
        plt.savefig(filename)
        print(f"Saved EDA plot: {filename}")

        if show_plots:
            plt.show()
        else:
            plt.close()

def visualize_missing_data(df: pd.DataFrame, stage: str, save_dir: str = "outputs/eda/imputed"):
    """
    Visualize missing data using missingno before and after imputation.

    Parameters:
    - df (pd.DataFrame): Dataset
    - stage (str): Label for the current stage (e.g., 'before', 'after_knn', 'after_mice')
    - save_dir (str): Directory path to save the plots
    """
    
    # Drop unwanted columns from visualization
    columns_to_exclude = ['Diagnostic Group', 'Diagnostic Group Code']
    df_plot = df.drop(columns=[col for col in columns_to_exclude if col in df.columns])

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    ax = msno.matrix(df_plot)
    plt.xticks(rotation=45)
    plt.title(f"Missing Data Matrix ({stage})", loc='center', fontsize=14, pad=20)
    plt.xlabel("Features")
    plt.ylabel("Records")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"missing_matrix_{stage}.png"))
    plt.close()

    print(f"Saved missing data plot for {stage}.")


