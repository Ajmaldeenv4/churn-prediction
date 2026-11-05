"""Streamlit dashboard — 4 pages: Overview, Model Comparison, Predict, Explain.

Reads persisted artifacts only; rendering a page never retrains or rewrites them.

Run from repo root or from `app/`:
    streamlit run app/app.py
    (or)  cd app && streamlit run app.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODELS = ROOT / "models"
ASSETS = APP_DIR / "assets"

sys.path.insert(0, str(ROOT / "src"))

from preprocess import INTERNET_ADD_ONS, encode_customer, load_raw  # noqa: E402

DEFAULT_TENURE = 12
DEFAULT_MONTHLY = 70.0
ADD_ON_LABELS = {
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
}

st.set_page_config(page_title="Churn Predictor", layout="wide", page_icon="📉")


@st.cache_resource
def load_model():
    with open(MODELS / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    scaler = joblib.load(MODELS / "scaler.pkl")
    with open(MODELS / "feature_names.json") as f:
        feature_names = json.load(f)
    return model, scaler, feature_names, shap.TreeExplainer(model)


@st.cache_data
def load_customers() -> pd.DataFrame:
    return load_raw()


@st.cache_data
def load_top_risk(n: int = 10) -> pd.DataFrame:
    """Highest-risk held-out customers, in original units."""
    holdout = pd.read_csv(MODELS / "holdout_predictions.csv")
    top = holdout.nlargest(n, "churn_probability").reset_index(drop=True)
    top["actual_churn"] = top["actual_churn"].map({1: "Yes", 0: "No"})
    top = top.rename(columns={
        "churn_probability": "Churn probability",
        "actual_churn": "Actually churned",
    })
    return top[["customerID", "Churn probability", "Actually churned", "tenure",
                "MonthlyCharges", "TotalCharges", "Contract", "InternetService"]]


def sync_total_charges() -> None:
    """Keep the total consistent with the sliders; it can still be overridden by hand."""
    st.session_state["total"] = float(st.session_state["tenure"] * st.session_state["monthly"])


def yes_no(label: str, key: str, default: str = "No") -> str:
    options = ["No", "Yes"]
    return st.radio(label, options, index=options.index(default), horizontal=True, key=key)


# -------- sidebar --------
page = st.sidebar.selectbox(
    "Navigate",
    ["Overview", "Model Comparison", "Predict", "Explain"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Telco Customer Churn — XGBoost + SHAP")

model, scaler, feature_names, explainer = load_model()
raw = load_customers()

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "Overview":
    st.title("📉 Telco Customer Churn — Overview")

    churned = raw["Churn"] == "Yes"
    contract_rates = churned.groupby(raw["Contract"]).mean()
    monthly_rate = contract_rates["Month-to-month"]
    two_year_rate = contract_rates["Two year"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(raw):,}")
    c2.metric("Features", "19 (pre-encoding)")
    c3.metric("Churn rate", f"{churned.mean():.1%}")

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
        f"- **Contract**: month-to-month customers churn at ~{monthly_rate / two_year_rate:.0f}× "
        f"the rate of 2-year customers ({monthly_rate:.1%} vs. {two_year_rate:.1%}).\n"
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
                 width="stretch")

    c1, c2 = st.columns(2)
    c1.image(str(ASSETS / "roc_curves.png"), caption="ROC curves on the test set")
    c2.image(str(ASSETS / "confusion_matrix.png"), caption="Confusion matrix — XGBoost")

    scores = comp.set_index("model")
    xgb = scores.loc["XGBoost"]
    lr = scores.loc["LogisticRegression"]
    st.success(
        f"**Served model: XGBoost** — test ROC-AUC {xgb['roc_auc']:.3f}, "
        f"5-fold CV ROC-AUC {xgb['cv_roc_auc_mean']:.3f}."
    )
    st.markdown(
        f"**Why XGBoost?** It has the highest CV and test ROC-AUC, but its lead over "
        f"Logistic Regression is small (CV {xgb['cv_roc_auc_mean']:.3f} vs. "
        f"{lr['cv_roc_auc_mean']:.3f}). It is served because it captures non-linear effects "
        "such as the jump in risk above $90 a month, and SHAP's TreeExplainer gives exact "
        "per-customer explanations for it. Hyperparameters were tuned with `GridSearchCV` "
        "over `n_estimators`, `max_depth`, and `learning_rate` using `scoring='roc_auc'`, "
        "with SMOTE and scaling refit inside every CV fold."
    )

    st.subheader("Does SMOTE help?")
    smote = pd.read_csv(MODELS / "smote_comparison.csv")
    st.dataframe(smote.style.format({c: "{:.4f}" for c in smote.columns if c != "strategy"}),
                 width="stretch")
    st.markdown(
        "Both rows are the tuned XGBoost on identical folds. SMOTE lifts recall at the 0.5 "
        "cut-off but does not improve ROC-AUC, and its Brier score is slightly worse. "
        f"`mean_predicted` should match the {(raw['Churn'] == 'Yes').mean():.1%} base rate. "
        "Plain SMOTE interpolates one-hot columns, so most synthetic rows contain fractional "
        "values no real customer has, which trees can isolate instead of learning general "
        "churn patterns."
    )

# ============================================================
# PAGE 3 — PREDICT
# ============================================================
elif page == "Predict":
    st.title("🔮 Predict churn risk")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Customer profile")
        tenure = st.slider("Tenure (months)", 0, 72, DEFAULT_TENURE,
                           key="tenure", on_change=sync_total_charges)
        monthly = st.slider("Monthly charges ($)", 18.0, 120.0, DEFAULT_MONTHLY,
                            key="monthly", on_change=sync_total_charges)
        st.session_state.setdefault("total", DEFAULT_TENURE * DEFAULT_MONTHLY)
        total = st.number_input("Total charges ($)", 0.0, 9000.0, key="total",
                                help="Follows tenure × monthly charges; edit to override.")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"],
                                key="contract")
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], key="internet")
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ], key="payment")
        phone = yes_no("Phone service", "phone", default="Yes")
        lines = st.selectbox("Multiple lines", ["No", "Yes"], key="multiple_lines",
                             disabled=phone == "No")
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True, key="gender")
        senior = yes_no("Senior citizen", "senior")
        partner = yes_no("Partner", "partner")
        dependents = yes_no("Dependents", "dependents")
        paperless = yes_no("Paperless billing", "paperless", default="Yes")
        with st.expander("Add-on services"):
            add_ons = {
                name: st.checkbox(ADD_ON_LABELS[name], key=name, disabled=internet == "No")
                for name in INTERNET_ADD_ONS
            }

    customer = dict(
        gender=gender, SeniorCitizen=int(senior == "Yes"), Partner=partner,
        Dependents=dependents, tenure=tenure, PhoneService=phone, MultipleLines=lines,
        InternetService=internet, Contract=contract, PaperlessBilling=paperless,
        PaymentMethod=payment, MonthlyCharges=monthly, TotalCharges=total,
        **{name: "Yes" if selected else "No" for name, selected in add_ons.items()},
    )
    features = encode_customer(customer, feature_names)
    scaled = scaler.transform(features)
    proba = float(model.predict_proba(scaled)[0, 1])

    with right:
        st.subheader("Result")
        st.metric("Churn probability", f"{proba:.1%}")
        if proba > 0.5:
            st.error(f"⚠️  High risk — {proba:.1%} probability of churn")
        else:
            st.success(f"✅  Low risk — {proba:.1%} probability of churn")

        st.subheader("Why this score")
        st.caption(
            "Each bar shows how far one feature moved this customer from the model's "
            "baseline, in log-odds. Red raises churn risk, blue lowers it."
        )
        scored = explainer(scaled)
        shap.plots.waterfall(
            shap.Explanation(
                values=scored.values[0], base_values=scored.base_values[0],
                data=features[0], feature_names=feature_names,
            ),
            max_display=10, show=False,
        )
        figure = plt.gcf()
        st.pyplot(figure)
        plt.close(figure)

    st.markdown("---")
    st.subheader("Top 10 highest-risk customers (held-out test set)")
    st.dataframe(load_top_risk(), width="stretch")

# ============================================================
# PAGE 4 — EXPLAIN
# ============================================================
elif page == "Explain":
    st.title("🔍 SHAP explanations")
    st.markdown(
        "**How to read these:** each feature's SHAP value is its contribution to "
        "moving the prediction *above* or *below* the baseline, in log-odds. The baseline "
        "is close to 0 (a 50% probability) because the model trains on SMOTE-balanced data. "
        "Red = pushes toward churn, blue = pushes away."
    )

    st.subheader("Global feature importance — beeswarm")
    st.image(str(ASSETS / "shap_summary.png"))
    st.markdown(
        "Each dot is one customer. Position on the x-axis is the SHAP value "
        "(impact on the model's output). Color is the feature's value — red = high, "
        "blue = low. **`Contract_Month-to-month`** (red dots on the right) and "
        "**`tenure`** (low values pushing right, high values pushing left) dominate."
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
        "The x-axis is `MonthlyCharges` in dollars and the y-axis is that feature's SHAP "
        "value for each customer. Its effect stays small between about $50 and $90, then "
        "climbs above about $90. Color splits customers by `Contract_Month-to-month`, "
        "SHAP's automatically chosen interaction feature."
    )
