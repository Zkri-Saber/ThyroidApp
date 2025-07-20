from setuptools import setup, find_packages

setup(
    name="thyroid_analysis",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "fancyimpute",
        "xgboost",
        "lightgbm",
        "catboost",
        "matplotlib",
        "seaborn",
        "matplotlib-venn",
    ],
    entry_points={
        'console_scripts': [
            'thyroid-run = main:main',
        ],
    },
)