"""Loading and preprocessing for the Student Performance dataset.

The dataset mixes numeric columns (age, absences, grades) with categorical
columns (school, parent's job, ...). We let scikit-learn handle both in one
ColumnTransformer so the exact same transformation is applied during
training and prediction.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "student-mat.csv"
TARGET = "G3"  # final grade, 0-20

# G1 and G2 are the period-1 and period-2 grades. Including them makes the
# prediction much easier (they correlate strongly with G3). The training
# script also reports metrics WITHOUT them so we can see how early we can
# flag at-risk students before any exam grades exist.
MIDTERM_FEATURES = ["G1", "G2"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV (semicolon-separated, as published by UCI)."""
    return pd.read_csv(path, sep=";")


def split_features_target(df: pd.DataFrame, drop_midterms: bool = False):
    """Return (X, y) where y is the final grade G3."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    if drop_midterms:
        X = X.drop(columns=MIDTERM_FEATURES)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """One-hot encode categoricals, standard-scale numerics."""
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )
