"""
Classification pipeline: compare models such as Logistic Regression, Random Forest,
and Gradient Boosted Trees on any CSV dataset. New models can be added in MODELS: dict

Usage:
    python classification_models.py --csv path/to/data.csv --target column_name
    python classification_models.py --csv path/to/data.csv --target column_name --test-size 0.25

"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


#Load dataset in csv
def load_csv(path: str, target: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available columns: {list(df.columns)}"
        )
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# Preprocessing data func

def preprocess(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    # Encode categorical features
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))

    # Encode target if it is categorical
    if y.dtype == object or str(y.dtype) == "category":
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = y.to_numpy()

    return X, y


#Model specs
MODELS: dict[str, Pipeline] = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]),
    "Gradient Boosted Trees": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ]),
}


# Evaluation func
def evaluate(
    name: str,
    model: Pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "y_pred": y_pred,
        "fitted_model": model,
    }


# Report results

def print_summary(results: list[dict], y_test: np.ndarray) -> None:
    print("\n" + "=" * 60)
    print(f"{'MODEL':<28} {'ACCURACY':>9} {'F1 (test)':>10} {'CV F1':>14}")
    print("-" * 60)
    for r in results:
        cv_str = f"{r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}"
        print(
            f"{r['model']:<28} {r['accuracy']:>9.4f} {r['f1_weighted']:>10.4f} {cv_str:>14}"
        )
    print("=" * 60)

    best = max(results, key=lambda r: r["f1_weighted"])
    print(f"\nBest model: {best['model']}  (F1 = {best['f1_weighted']:.4f})\n")

    print("--- Classification report for best model ---")
    print(classification_report(y_test, best["y_pred"]))


# Plot results
def plot_results(results: list[dict], y_test: np.ndarray) -> None:
    model_names = [r["model"] for r in results]
    metrics = pd.DataFrame({
        "Model": model_names * 2,
        "Score": [r["accuracy"] for r in results] + [r["f1_weighted"] for r in results],
        "Metric": ["Accuracy"] * len(results) + ["F1 Weighted"] * len(results),
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Classification Pipeline — Model Comparison", fontsize=14, fontweight="bold")

    # Bar chart: accuracy & F1 side by side
    sns.barplot(data=metrics, x="Model", y="Score", hue="Metric", ax=axes[0], palette="muted")
    axes[0].set_title("Accuracy vs F1 Weighted")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)

    # Confusion matrix for the best model
    best = max(results, key=lambda r: r["f1_weighted"])
    cm = confusion_matrix(y_test, best["y_pred"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[1],
        linewidths=0.5,
    )
    axes[1].set_title(f"Confusion Matrix — {best['model']}")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()


# Parser with args
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CSV classification pipeline")
    p.add_argument("--csv", required=True, help="Path to CSV file")
    p.add_argument("--target", required=True, help="Name of the target column")
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data held out for testing (default: 0.2)",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    return p


def run(csv_path: str, target: str, test_size: float = 0.2, cv_folds: int = 5) -> list[dict]:
    print(f"\nLoading data from: {csv_path}")
    X, y = load_csv(csv_path, target)
    print(f"Shape: {X.shape}  |  Classes: {y.nunique() if hasattr(y, 'nunique') else len(set(y))}")

    X, y = preprocess(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples\n")

    results = []
    for name, model in MODELS.items():
        print(f"Training {name}...")
        r = evaluate(name, model, X_train, X_test, y_train, y_test, cv_folds)
        results.append(r)

    print_summary(results, y_test)
    plot_results(results, y_test)
    return results


def main() -> None:
    args = build_parser().parse_args()
    run(args.csv, args.target, args.test_size, args.cv_folds)


if __name__ == "__main__":
    main()