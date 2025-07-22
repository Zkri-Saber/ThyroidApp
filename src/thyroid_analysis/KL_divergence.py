# src/thyroid_analysis/KL_divergence.py

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

def calculate_kl_divergence(original, imputed, bins=20):
    """
    Calculate KL divergence between original and imputed data using histogram bins.
    Adds a small epsilon to avoid log(0).
    """
    original = original.dropna()
    imputed = imputed.dropna()
    if original.empty or imputed.empty:
        return 0.0

    min_val = min(original.min(), imputed.min())
    max_val = max(original.max(), imputed.max())

    orig_hist, bin_edges = np.histogram(original, bins=bins, range=(min_val, max_val), density=True)
    imputed_hist, _ = np.histogram(imputed, bins=bin_edges, density=True)

    orig_hist += 1e-10
    imputed_hist += 1e-10

    return entropy(orig_hist, imputed_hist)

def compute_kl_for_all_features(df_original, df_knn, df_mice, columns):
    """
    Compute KL divergence for each feature for both KNN and MICE.
    Returns a DataFrame for plotting and export.
    """
    kl_knn = []
    kl_mice = []

    for col in columns:
        kl_knn.append(calculate_kl_divergence(df_original[col], df_knn[col]))
        kl_mice.append(calculate_kl_divergence(df_original[col], df_mice[col]))

    return pd.DataFrame({
        'Feature': columns,
        'KL(KNN)': kl_knn,
        'KL(MICE)': kl_mice
    })

def plot_kl_divergence(kl_df, save_path="outputs/eda/imputed/kl_divergence_plot.png"):
    """
    Plot and save a bar chart of KL divergence values.
    """
    kl_long = kl_df.melt(id_vars='Feature', var_name='Method', value_name='KL Divergence')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=kl_long, x='Feature', y='KL Divergence', hue='Method')
    plt.xticks(rotation=45, ha='right')
    plt.title("KL Divergence: KNN vs MICE Imputation")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 KL divergence plot saved to: {save_path}")
