"""Train LogisticRegression, RandomForest, and XGBoost on the Telco data.

Outputs:
  - models/xgb_model.pkl       best (tuned) XGBoost estimator
  - models/comparison.csv      per-model metrics
  - app/assets/roc_curves.png  overlaid ROC curves
  - app/assets/confusion_matrix.png  confusion matrix for best model
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier

from preprocess import prepare_data

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ASSETS = ROOT / "app" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def evaluate(name, model, X_test, y_test, cv_score):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "cv_roc_auc_mean": cv_score,
    }, proba, pred


def tune_xgb(X_train, y_train):
    base = XGBClassifier(
        eval_metric="logloss", random_state=RANDOM_STATE,
        n_jobs=-1, tree_method="hist",
    )
    grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
    }
    search = GridSearchCV(base, grid, scoring="roc_auc", cv=5, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    print(f"  best params: {search.best_params_}")
    print(f"  best cv roc_auc: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_score_


def main() -> None:
    print("Loading data...")
    X_train, X_test, y_train, y_test, _ = prepare_data()

    models = {}
    cv_scores = {}

    print("\n[1/3] Logistic Regression")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    cv_scores["LogisticRegression"] = cross_val_score(
        lr, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
    ).mean()
    lr.fit(X_train, y_train)
    models["LogisticRegression"] = lr

    print("\n[2/3] Random Forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    cv_scores["RandomForest"] = cross_val_score(
        rf, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
    ).mean()
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    print("\n[3/3] XGBoost (GridSearchCV)")
    xgb, xgb_cv = tune_xgb(X_train, y_train)
    cv_scores["XGBoost"] = xgb_cv
    models["XGBoost"] = xgb

    rows = []
    probas = {}
    for name, mdl in models.items():
        row, proba, _ = evaluate(name, mdl, X_test, y_test, cv_scores[name])
        rows.append(row)
        probas[name] = proba
        print(f"\n{name}: acc={row['accuracy']:.3f}  f1={row['f1']:.3f}  "
              f"roc_auc={row['roc_auc']:.3f}  cv_roc_auc={row['cv_roc_auc_mean']:.3f}")

    comparison = pd.DataFrame(rows).round(4)
    comparison.to_csv(MODELS_DIR / "comparison.csv", index=False)
    print(f"\nSaved {MODELS_DIR / 'comparison.csv'}")
    print(comparison.to_string(index=False))

    plt.figure(figsize=(7, 6))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves — test set")
    plt.legend(loc="lower right")
    plt.savefig(ASSETS / "roc_curves.png", bbox_inches="tight", dpi=150)
    plt.clf()

    best_name = comparison.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    print(f"\nBest model by test ROC-AUC: {best_name}")
    best_pred = (probas[best_name] >= 0.5).astype(int)
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"])
    plt.title(f"Confusion matrix — {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(ASSETS / "confusion_matrix.png", bbox_inches="tight", dpi=150)
    plt.clf()

    with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(models["XGBoost"], f)
    print(f"Saved {MODELS_DIR / 'xgb_model.pkl'}")


if __name__ == "__main__":
    main()
