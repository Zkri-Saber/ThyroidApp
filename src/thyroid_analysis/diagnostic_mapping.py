# src/thyroid_analysis/diagnostic_mapping.py

# Mapping raw diagnosis text to simplified diagnostic groups
diagnostic_mapping = {
    'No Disease': 'No Disease',
    'Hyperthyroidisim': 'Hyperthyroidism',
    'Hyperthyroidisim, Multinodular Goiter (MNG)': 'Hyperthyroidism',
    'Graves Disease (GD), Hyperthyroidisim': 'Hyperthyroidism',
    'Hyperthyroidism, Multinodular Goiter (MNG)': 'Hyperthyroidism',
    'Hyperthyroidism': 'Hyperthyroidism',
    'Hyperthyroidisim, Thyroid Nodule': 'Hyperthyroidism',
    'hyperthyroid': 'Hyperthyroidism',
    'Graves Disease (GD), Hyperthyroidism': 'Hyperthyroidism',
    'Hyperthyroidism, Suspicious Thyroid Nodule': 'Hyperthyroidism',
    'hyper for 2 ys': 'Hyperthyroidism',
    'hyperthyroid for 15 month': 'Hyperthyroidism',
    'hyperthyroid for  3 ys': 'Hyperthyroidism',
    'hyperthyroid for 6 ys': 'Hyperthyroidism',

    'euthyroid': 'Euthyroid',
    'Euthyroid, Thyroid Nodule': 'Euthyroid',
    'Euthyroid, Papillary Thyroid Carcinoma (PTC)': 'Euthyroid',
    'Euthyroid, Suspicious Thyroid Nodule': 'Euthyroid',
    'Euthyroid, Multinodular Goiter (MNG)': 'Euthyroid',
    'Euthyroid, Multinodular Goiter (MNG), Suspicious Thyroid Nodule': 'Euthyroid',
    'Euthyroid, Medullary Thyroid Carcinoma': 'Euthyroid',
    'Euthyroid, Parathryoid Adenoma': 'Euthyroid',
    'Euthyroid, Papillary Thyroid Carcinoma (PTC), Suspicious Thyroid Nodule': 'Euthyroid',

    'hypothyroid': 'Hypothyroidism',
    'Hypothyroidism, Suspicious Thyroid Nodule': 'Hypothyroidism',
    'Hypothyroidism, Papillary Thyroid Carcinoma (PTC)': 'Hypothyroidism',
    'Hypothyroidism, Thyroid Nodule': 'Hypothyroidism',
    'Hypothyroidism, Multinodular Goiter (MNG)': 'Hypothyroidism',
    'Hypothyroidism, Multinodular Goiter (MNG), Papillary Thyroid Carcinoma (PTC)': 'Hypothyroidism',
    'Hypothyroidism, RSE': 'Hypothyroidism',
    'Hypothyroidism, Papillary Thyroid Microcarcinoma': 'Hypothyroidism',
    'Hypothyroidism, Papillary Thyroid Carcinoma (PTC), Positive Cervical LN': 'Hypothyroidism',
    'Chronic Thyroiditis, Hypothyroidism': 'Hypothyroidism',
    'Hypoparathyroidism, Hypothyroidism': 'Hypothyroidism',
    'Hypothyroidism, Multinodular Goiter (MNG), Suspicious Thyroid Nodule': 'Hypothyroidism',
    'Hypothyroidism, Papillary Thyroid Carcinoma (PTC), Thyroid Nodule': 'Hypothyroidism',
    'Hypoparathyroidism, Hypothyroidism, Papillary Thyroid Carcinoma (PTC)': 'Hypothyroidism',
}

# Mapping diagnostic group to numerical code
diagnostic_group_mapping = {
    'No Disease': 0,
    'Hyperthyroidism': 1,
    'Euthyroid': 2,
    'Hypothyroidism': 3
}
