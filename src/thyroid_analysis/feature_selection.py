# src/thyroid_analysis/feature_selection.py

from collections import Counter
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

def select_features_consensus(X, y, numerical_columns, n_feat=8, n_pc=5):
    """
    Implements RO_2 FEO as shown in the diagram:
    - Uses RFE, PCA, and DT to select important features.
    - Aggregates them into Local/Global CFS.
    - Produces Ensemble Biomarkers via majority voting.
    
    Parameters:
    - X (pd.DataFrame): Feature matrix
    - y (pd.Series): Target labels
    - numerical_columns (list): List of numerical features
    - n_feat (int): Top features from each technique
    - n_pc (int): Number of principal components (for PCA)
    
    Returns:
    - dict: {
        'RFE': [...],
        'PCA': [...],
        'DecisionTree': [...],
        'Consensus': [...],  # Final biomarkers
    }
    """

    # === Step 1: RFE ===
    rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=n_feat)
    rfe.fit(X[numerical_columns], y)
    F_rfe = [feat for feat, sel in zip(numerical_columns, rfe.support_) if sel]

    # === Step 2: Decision Tree Importance ===
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X[numerical_columns], y)
    importances = dt.feature_importances_
    F_dt = [f for f, _ in sorted(zip(numerical_columns, importances), key=lambda x: x[1], reverse=True)[:n_feat]]

    # === Step 3: PCA Top Loadings ===
    pca = PCA(n_components=n_pc)
    pca.fit(X[numerical_columns])
    loadings = abs(pca.components_[0])
    F_pca = [f for f, _ in sorted(zip(numerical_columns, loadings), key=lambda x: x[1], reverse=True)[:n_feat]]

    # === Step 4: Consensus Voting (>=2 votes)
    all_feats = F_rfe + F_dt + F_pca
    counts = Counter(all_feats)
    F_consensus = [f for f, c in counts.items() if c >= 2]

    return {
        "RFE": F_rfe,
        "DecisionTree": F_dt,
        "PCA": F_pca,
        "Consensus": F_consensus
    }
