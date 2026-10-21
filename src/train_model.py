"""Train the two CampusPulse models and save them with their metrics.

Model 1 - grade regressor: predicts the final grade G3 (0-20).
Model 2 - at-risk classifier: predicts pass (G3 >= 10) vs fail.

Both share the same preprocessing pipeline, so a saved model is a full
Pipeline object: raw student record in, prediction out.
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import build_preprocessor, load_data, split_features_target

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
PASS_MARK = 10  # G3 >= 10 counts as a pass on the 0-20 Portuguese scale


def make_pipeline(X: pd.DataFrame, model) -> Pipeline:
    """Preprocessing + model in one Pipeline, so prediction needs raw input."""
    return Pipeline([("preprocess", build_preprocessor(X)), ("model", model)])


def regression_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(rmse, 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }


def train_regressor(X_train, X_test, y_train, y_test):
    model = make_pipeline(X_train, GradientBoostingRegressor(random_state=42))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    metrics = regression_metrics(y_test, y_pred)
    metrics["cv_r2_mean"] = round(float(cv.mean()), 3)
    return model, y_pred, metrics


def train_classifier(X_train, X_test, y_train, y_test):
    y_train_bin = (y_train >= PASS_MARK).astype(int)
    y_test_bin = (y_test >= PASS_MARK).astype(int)
    model = make_pipeline(X_train, GradientBoostingClassifier(random_state=42))
    model.fit(X_train, y_train_bin)
    y_pred = model.predict(X_test)
    cv = cross_val_score(model, X_train, y_train_bin, cv=5, scoring="f1")
    metrics = {
        "accuracy": round(float(accuracy_score(y_test_bin, y_pred)), 3),
        "f1": round(float(f1_score(y_test_bin, y_pred)), 3),
        "cv_f1_mean": round(float(cv.mean()), 3),
    }
    return model, y_test_bin, y_pred, metrics


def save_figures(df, regressor, X_test, y_test, y_pred, clf_y_test, clf_y_pred,
                 feature_names, importances):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Distribution of final grades
    plt.figure(figsize=(8, 5))
    sns.histplot(df["G3"], bins=21, kde=True, color="#4C72B0")
    plt.title("Distribution of final grades (G3)")
    plt.xlabel("Final grade (0-20)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "grade_distribution.png", dpi=150)
    plt.close()

    # 2. Correlation heatmap of numeric features
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(exclude="object").corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation between numeric features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # 3. Actual vs predicted grades
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="#4C72B0", edgecolor="white")
    plt.plot([0, 20], [0, 20], "r--", label="Perfect prediction")
    plt.xlabel("Actual final grade")
    plt.ylabel("Predicted final grade")
    plt.title("Actual vs predicted final grades (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # 4. Confusion matrix of the at-risk classifier
    cm = confusion_matrix(clf_y_test, clf_y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("At-risk classifier (test set)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # 5. Top-15 feature importances of the grade regressor
    order = np.argsort(importances)[-15:]
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[order], importances[order], color="#4C72B0")
    plt.xlabel("Importance")
    plt.title("Top 15 features driving grade prediction")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close()


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} students with {df.shape[1]} columns")

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    regressor, y_pred, reg_metrics = train_regressor(X_train, X_test, y_train, y_test)
    classifier, clf_y_test, clf_y_pred, clf_metrics = train_classifier(
        X_train, X_test, y_train, y_test
    )

    # Early-warning variant: how well can we predict WITHOUT midterm grades?
    X_early = X.drop(columns=["G1", "G2"])
    Xe_train, Xe_test, ye_train, ye_test = train_test_split(
        X_early, y, test_size=0.2, random_state=42
    )
    _, _, early_metrics = train_regressor(Xe_train, Xe_test, ye_train, ye_test)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(regressor, MODELS_DIR / "grade_regressor.joblib")
    joblib.dump(classifier, MODELS_DIR / "atrisk_classifier.joblib")

    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "features": int(X.shape[1]),
            "test_size": 0.2,
            "pass_mark": PASS_MARK,
        },
        "grade_regressor": reg_metrics,
        "grade_regressor_no_midterms": early_metrics,
        "atrisk_classifier": clf_metrics,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    # Feature importances from the fitted regressor (after one-hot encoding)
    fitted = regressor.named_steps["model"]
    feature_names = regressor.named_steps["preprocess"].get_feature_names_out()
    save_figures(df, regressor, X_test, y_test, y_pred, clf_y_test, clf_y_pred,
                 feature_names, fitted.feature_importances_)

    print(json.dumps(metrics, indent=2))
    print(f"Models saved to {MODELS_DIR}")
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
