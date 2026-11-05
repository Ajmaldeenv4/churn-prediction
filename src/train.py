"""Train LogisticRegression, RandomForest, and XGBoost on the Telco data.

Cross-validation runs on the original training fold with SMOTE and scaling inside
each fold, so no synthetic row can appear on both sides of a CV split.

Outputs:
  - models/xgb_model.pkl             tuned XGBoost, the model the dashboard serves
  - models/comparison.csv            per-model metrics
  - models/smote_comparison.csv      tuned XGBoost with vs. without SMOTE
  - models/holdout_predictions.csv   test-set scores in original units
  - app/assets/roc_curves.png        overlaid ROC curves
  - app/assets/confusion_matrix.png  confusion matrix for the served model
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocess import load_raw, prepare_data, split_data

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ASSETS = ROOT / "app" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
XGB_BASE = dict(eval_metric="logloss", random_state=RANDOM_STATE, tree_method="hist")
HOLDOUT_COLUMNS = [
    "customerID", "tenure", "MonthlyCharges", "TotalCharges",
    "Contract", "InternetService", "PaymentMethod",
]


def smote_pipeline(classifier) -> Pipeline:
    """Resample and scale inside whichever fold is being fit."""
    return Pipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("scale", StandardScaler()),
        ("clf", classifier),
    ])


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
    }, proba


def tune_xgb(X_train, y_train):
    """Return (best params, best CV ROC-AUC), searched on the original training fold."""
    grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.01, 0.05, 0.1],
    }
    pipeline = smote_pipeline(XGBClassifier(**XGB_BASE, n_jobs=1))
    search = GridSearchCV(
        pipeline, grid, scoring="roc_auc", cv=CV, n_jobs=-1, error_score="raise"
    )
    search.fit(X_train, y_train)
    params = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
    print(f"  best params: {params}")
    print(f"  best cv roc_auc: {search.best_score_:.4f}")
    return params, search.best_score_


def compare_smote(params, X_train, y_train) -> pd.DataFrame:
    """Score the tuned XGBoost with and without SMOTE on identical folds."""
    classifier = XGBClassifier(**XGB_BASE, **params, n_jobs=1)
    plain = Pipeline([("scale", StandardScaler()), ("clf", classifier)])
    rows = []
    for name, pipeline in [("No resampling", plain),
                           ("SMOTE in each fold", smote_pipeline(classifier))]:
        proba = cross_val_predict(
            pipeline, X_train, y_train, cv=CV, method="predict_proba", n_jobs=-1
        )[:, 1]
        rows.append({
            "strategy": name,
            "cv_roc_auc": roc_auc_score(y_train, proba),
            "cv_brier": brier_score_loss(y_train, proba),
            "cv_recall": recall_score(y_train, proba >= 0.5),
            "mean_predicted": proba.mean(),
        })
    return pd.DataFrame(rows).round(4)


def save_holdout(test_index, y_test, proba) -> None:
    """Persist test-set scores in original units for the dashboard to read."""
    holdout = load_raw().loc[test_index, HOLDOUT_COLUMNS].assign(
        actual_churn=y_test, churn_probability=proba.round(4)
    )
    holdout.sort_values("churn_probability", ascending=False).to_csv(
        MODELS_DIR / "holdout_predictions.csv", index=False
    )


def main() -> None:
    print("Loading data...")
    X_train_raw, _, y_train_raw, _, _, test_index = split_data()
    X_train, X_test, y_train, y_test, _ = prepare_data()

    print("\n[1/3] Logistic Regression")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_cv = cross_val_score(
        smote_pipeline(lr), X_train_raw, y_train_raw, cv=CV, scoring="roc_auc", n_jobs=-1
    ).mean()

    print("\n[2/3] Random Forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_cv = cross_val_score(
        smote_pipeline(rf), X_train_raw, y_train_raw, cv=CV, scoring="roc_auc", n_jobs=-1
    ).mean()

    print("\n[3/3] XGBoost (GridSearchCV)")
    xgb_params, xgb_cv = tune_xgb(X_train_raw, y_train_raw)
    xgb = XGBClassifier(**XGB_BASE, **xgb_params, n_jobs=-1)

    models = {"LogisticRegression": lr, "RandomForest": rf, "XGBoost": xgb}
    cv_scores = {"LogisticRegression": lr_cv, "RandomForest": rf_cv, "XGBoost": xgb_cv}

    rows = []
    probas = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        row, proba = evaluate(name, mdl, X_test, y_test, cv_scores[name])
        rows.append(row)
        probas[name] = proba
        print(f"\n{name}: acc={row['accuracy']:.3f}  f1={row['f1']:.3f}  "
              f"roc_auc={row['roc_auc']:.3f}  cv_roc_auc={row['cv_roc_auc_mean']:.3f}")

    comparison = pd.DataFrame(rows).round(4)
    comparison.to_csv(MODELS_DIR / "comparison.csv", index=False)
    print(f"\nSaved {MODELS_DIR / 'comparison.csv'}")
    print(comparison.to_string(index=False))

    smote = compare_smote(xgb_params, X_train_raw, y_train_raw)
    smote.to_csv(MODELS_DIR / "smote_comparison.csv", index=False)
    print(f"\nSaved {MODELS_DIR / 'smote_comparison.csv'}")
    print(smote.to_string(index=False))

    save_holdout(test_index, y_test, probas["XGBoost"])

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

    cm = confusion_matrix(y_test, (probas["XGBoost"] >= 0.5).astype(int))
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"])
    plt.title("Confusion matrix — XGBoost")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(ASSETS / "confusion_matrix.png", bbox_inches="tight", dpi=150)
    plt.clf()

    with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(models["XGBoost"], f)
    print(f"Saved {MODELS_DIR / 'xgb_model.pkl'}")


if __name__ == "__main__":
    main()
