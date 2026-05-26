"""Helper for the Streamlit app: build the top-10 highest-risk table from the
held-out test set without re-running preprocess.prepare_data each render.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import prepare_data  # noqa: E402


@st.cache_data
def top_risk_table(n: int = 10) -> pd.DataFrame:
    _, X_test, _, y_test, feature_names = prepare_data()
    with open(ROOT / "models" / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    proba = model.predict_proba(X_test)[:, 1]
    df = pd.DataFrame(X_test, columns=feature_names)
    df.insert(0, "actual_churn", y_test)
    df.insert(0, "predicted_proba", proba)
    df = df.sort_values("predicted_proba", ascending=False).head(n).reset_index(drop=True)
    display_cols = ["predicted_proba", "actual_churn", "tenure",
                    "MonthlyCharges", "TotalCharges",
                    "Contract_Month-to-month", "InternetService_Fiber optic"]
    return df[display_cols].round(3)
