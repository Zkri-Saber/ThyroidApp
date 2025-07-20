import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from .config import RFE_FEATURE_COUNT, PCA_COMPONENTS, RANDOM_FOREST_PARAMS

def select_by_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    estimator=None,
    n_features=None
) -> pd.DataFrame:
    estimator = estimator or DecisionTreeClassifier(**RANDOM_FOREST_PARAMS)
    n_features = n_features or RFE_FEATURE_COUNT
    selector = RFE(estimator, n_features_to_select=n_features)
    selector.fit(X, y)
    return X.loc[:, selector.support_]

def select_by_pca(
    X: pd.DataFrame,
    n_components=None
) -> pd.DataFrame:
    n_components = n_components or PCA_COMPONENTS
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)
    cols = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(components, columns=cols)
