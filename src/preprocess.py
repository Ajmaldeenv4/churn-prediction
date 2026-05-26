"""Preprocessing pipeline for the Telco churn dataset.

Run directly to materialize artifacts (scaler.pkl, feature_names.json) for
downstream `train.py` / `explain.py`, or import `prepare_data()` for the arrays.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

BINARY_YES_NO = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
ONE_HOT_COLS = ["InternetService", "Contract", "PaymentMethod", "MultipleLines"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df = df.drop(columns=["customerID"])
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Treat "No internet service" / "No phone service" as 0 alongside "No".
    for col in BINARY_YES_NO:
        df[col] = (df[col] == "Yes").astype(int)
    df["gender"] = (df["gender"] == "Male").astype(int)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=False)
    # Ensure no boolean dtypes leak into XGBoost / scaler.
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def prepare_data(random_state: int = 42, test_size: float = 0.2):
    """Return (X_train_res, X_test, y_train_res, y_test, feature_names).

    Side effects: writes `models/scaler.pkl` and `models/feature_names.json`.

    Critical rules:
      - SMOTE is applied ONLY on the training fold.
      - StandardScaler is fit ONLY on the (SMOTE-resampled) training fold.
    """
    df = encode(load_raw())
    y = df["Churn"].values
    X = df.drop(columns=["Churn"])
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, stratify=y, random_state=random_state
    )

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    return X_train_res, X_test, y_train_res, y_test, feature_names


if __name__ == "__main__":
    X_train_res, X_test, y_train_res, y_test, feats = prepare_data()
    print(f"X_train_res: {X_train_res.shape}  y_train_res balance: {np.bincount(y_train_res)}")
    print(f"X_test:      {X_test.shape}  y_test balance: {np.bincount(y_test)}")
    print(f"features ({len(feats)}): {feats}")
    print(f"Saved scaler -> {MODELS_DIR / 'scaler.pkl'}")
    print(f"Saved feature_names -> {MODELS_DIR / 'feature_names.json'}")
