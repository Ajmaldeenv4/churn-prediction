"""Streamlit dashboard — 4 pages: Overview, Model Comparison, Predict, Explain.

Run from repo root or from `app/`:
    streamlit run app/app.py
    (or)  cd app && streamlit run app.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODELS = ROOT / "models"
ASSETS = APP_DIR / "assets"
DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

st.set_page_config(page_title="Churn Predictor", layout="wide", page_icon="📉")


@st.cache_resource
def load_model():
    with open(MODELS / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    scaler = joblib.load(MODELS / "scaler.pkl")
    with open(MODELS / "feature_names.json") as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


@st.cache_data
def load_raw():
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


def build_feature_row(inputs: dict, feature_names: list[str]) -> np.ndarray:
    """Map raw form inputs to the model's encoded feature vector."""
    row = {name: 0 for name in feature_names}
    row["gender"] = 1 if inputs["gender"] == "Male" else 0
    row["SeniorCitizen"] = int(inputs["SeniorCitizen"])
    row["Partner"] = int(inputs["Partner"])
    row["Dependents"] = int(inputs["Dependents"])
    row["tenure"] = inputs["tenure"]
    row["PhoneService"] = int(inputs["PhoneService"])
    row["PaperlessBilling"] = int(inputs["PaperlessBilling"])
    row["MonthlyCharges"] = inputs["MonthlyCharges"]
    row["TotalCharges"] = inputs["TotalCharges"]
    for s in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]:
        row[s] = int(inputs[s])
    row[f"InternetService_{inputs['InternetService']}"] = 1
    row[f"Contract_{inputs['Contract']}"] = 1
    row[f"PaymentMethod_{inputs['PaymentMethod']}"] = 1
    row[f"MultipleLines_{inputs['MultipleLines']}"] = 1
    return np.array([[row[c] for c in feature_names]], dtype=float)


# -------- sidebar --------
page = st.sidebar.selectbox(
    "Navigate",
    ["Overview", "Model Comparison", "Predict", "Explain"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Telco Customer Churn — XGBoost + SHAP")

model, scaler, feature_names = load_model()
raw = load_raw()

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "Overview":
    st.title("📉 Telco Customer Churn — Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(raw):,}")
    c2.metric("Features", "20 (pre-encoding)")
    churn_rate = (raw["Churn"] == "Yes").mean()
    c3.metric("Churn rate", f"{churn_rate:.1%}")

    st.markdown(
        "Roughly **1 in 4 customers churn**. Any model that always predicts "
        "*no churn* still gets ~74% accuracy — so we balance the training set "
        "with SMOTE and watch recall / ROC-AUC instead of raw accuracy."
    )

    st.image(str(ASSETS / "churn_donut.png"), caption="Churn distribution", width=420)

    st.subheader("Categorical drivers")
    st.image(str(ASSETS / "churn_by_category.png"),
             caption="Churn rate by Contract, InternetService, PaymentMethod, SeniorCitizen")
    st.markdown(
        "- **Contract**: month-to-month customers churn ~3× more than 2-year customers.\n"
        "- **InternetService**: fiber-optic users churn more than DSL; 'No internet' churns least.\n"
        "- **PaymentMethod**: electronic-check users churn most.\n"
        "- **SeniorCitizen**: nearly 2× churn vs. non-seniors."
    )

    st.subheader("Numeric distributions")
    st.image(str(ASSETS / "numeric_distributions.png"),
             caption="tenure, MonthlyCharges, TotalCharges — churners vs non-churners")
    st.markdown(
        "Churners are concentrated in the **first 12 months** and at **higher monthly bills**. "
        "Long-tenured customers are durably stickier."
    )

    st.subheader("Correlations")
    st.image(str(ASSETS / "correlation_heatmap.png"), width=550)

# ============================================================
# PAGE 2 — MODEL COMPARISON
# ============================================================
elif page == "Model Comparison":
    st.title("🏁 Model Comparison")
    comp = pd.read_csv(MODELS / "comparison.csv")
    st.dataframe(comp.style.format({c: "{:.4f}" for c in comp.columns if c != "model"}),
                 use_container_width=True)

    c1, c2 = st.columns(2)
    c1.image(str(ASSETS / "roc_curves.png"), caption="ROC curves on the test set")
    c2.image(str(ASSETS / "confusion_matrix.png"), caption="Confusion matrix — XGBoost")

    best = comp.sort_values("roc_auc", ascending=False).iloc[0]
    st.success(
        f"**Selected: XGBoost** — test ROC-AUC {best['roc_auc']:.3f}, "
        f"5-fold CV ROC-AUC {best['cv_roc_auc_mean']:.3f}."
    )
    st.markdown(
        "**Why XGBoost?** Handles mixed feature types, robust to outliers, built-in "
        "L1/L2 regularization, and consistently the strongest CV ROC-AUC. "
        "Hyperparameters were tuned with `GridSearchCV` over `n_estimators`, "
        "`max_depth`, and `learning_rate` using `scoring='roc_auc'`."
    )

# ============================================================
# PAGE 3 — PREDICT
# ============================================================
elif page == "Predict":
    st.title("🔮 Predict churn risk")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Customer profile")
        with st.form("predict"):
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0)
            total = st.number_input("Total charges ($)", 0.0, 9000.0,
                                    value=float(monthly * max(tenure, 1)))
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            payment = st.selectbox("Payment method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            mult = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
            senior = st.radio("Senior citizen", [0, 1], horizontal=True)
            partner = st.radio("Partner", [0, 1], horizontal=True)
            dependents = st.radio("Dependents", [0, 1], horizontal=True)
            phone = st.radio("Phone service", [0, 1], index=1, horizontal=True)
            paperless = st.radio("Paperless billing", [0, 1], index=1, horizontal=True)
            with st.expander("Add-on services"):
                addons = {s: st.checkbox(s, value=False) for s in [
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies",
                ]}
            submitted = st.form_submit_button("Predict", use_container_width=True)

    with right:
        st.subheader("Result")
        if submitted:
            inputs = dict(
                gender=gender, SeniorCitizen=senior, Partner=partner,
                Dependents=dependents, tenure=tenure, PhoneService=phone,
                PaperlessBilling=paperless, MonthlyCharges=monthly,
                TotalCharges=total, InternetService=internet, Contract=contract,
                PaymentMethod=payment, MultipleLines=mult, **addons,
            )
            X_row = build_feature_row(inputs, feature_names)
            X_scaled = scaler.transform(X_row)
            proba = float(model.predict_proba(X_scaled)[0, 1])
            st.metric("Churn probability", f"{proba:.1%}")
            if proba > 0.5:
                st.error(f"⚠️  High risk — {proba:.1%} probability of churn")
            else:
                st.success(f"✅  Low risk — {proba:.1%} probability of churn")
        else:
            st.info("Fill the form and click Predict.")

        st.markdown("---")
        st.subheader("Top 10 highest-risk customers (held-out test set)")
        from preprocess_loader import top_risk_table  # type: ignore  # lazy import
        st.dataframe(top_risk_table(), use_container_width=True)

# ============================================================
# PAGE 4 — EXPLAIN
# ============================================================
elif page == "Explain":
    st.title("🔍 SHAP explanations")
    st.markdown(
        "**How to read these:** each feature's SHAP value is its contribution to "
        "moving the prediction *above* or *below* the baseline (the model's "
        "average predicted log-odds). Red = pushes toward churn, blue = pushes away."
    )

    st.subheader("Global feature importance — beeswarm")
    st.image(str(ASSETS / "shap_summary.png"))
    st.markdown(
        "Each dot is one customer. Position on the x-axis is the SHAP value "
        "(impact on the model's output). Color is the feature's raw value — "
        "red = high, blue = low. **`Contract_Two year` (low → red on the left)** "
        "and **`tenure` (low values pushing right)** dominate."
    )

    st.subheader("Mean |SHAP| ranking")
    st.image(str(ASSETS / "shap_bar.png"))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Why this customer churned")
        st.image(str(ASSETS / "shap_waterfall_churner.png"))
    with c2:
        st.subheader("Why this customer stayed")
        st.image(str(ASSETS / "shap_waterfall_nonchurner.png"))

    st.subheader("Dependence — MonthlyCharges")
    st.image(str(ASSETS / "shap_dependence_monthly.png"))
    st.markdown(
        "Reading the dependence plot: x-axis is the (scaled) value of `MonthlyCharges`; "
        "y-axis is that feature's SHAP value for each customer. The color reveals "
        "an interaction effect — typically with `tenure` or `Contract`."
    )
