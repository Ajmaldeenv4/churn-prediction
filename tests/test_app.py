"""Drive every dashboard page headlessly with Streamlit's AppTest."""
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "app.py"
MODELS = ROOT / "models"


def open_page(name: str) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    if name != "Overview":
        app.sidebar.selectbox[0].select(name).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def markdown_text(app: AppTest) -> str:
    return "\n".join(m.value for m in app.markdown)


@pytest.mark.parametrize("page", ["Overview", "Model Comparison", "Predict", "Explain"])
def test_every_page_renders_without_error(page):
    open_page(page)


def test_overview_counts_only_real_input_features():
    # 21 columns minus customerID and the Churn target.
    app = open_page("Overview")
    assert app.metric[1].value.startswith("19")


def test_overview_states_the_measured_contract_gap():
    raw = pd.read_csv(ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    rates = raw.assign(churned=raw["Churn"] == "Yes").groupby("Contract")["churned"].mean()
    ratio = rates["Month-to-month"] / rates["Two year"]
    assert f"~{ratio:.0f}×" in markdown_text(open_page("Overview"))


def test_model_comparison_headline_reports_the_served_model():
    comparison = pd.read_csv(MODELS / "comparison.csv").set_index("model")
    app = open_page("Model Comparison")
    headline = app.success[0].value
    assert "XGBoost" in headline
    assert f"{comparison.loc['XGBoost', 'roc_auc']:.3f}" in headline
    assert f"{comparison.loc['XGBoost', 'cv_roc_auc_mean']:.3f}" in headline


def test_model_comparison_shows_the_smote_evidence():
    app = open_page("Model Comparison")
    assert len(app.dataframe) == 2


def test_predict_scores_the_default_customer_without_a_submit_step():
    app = open_page("Predict")
    assert app.metric[0].label == "Churn probability"
    assert app.metric[0].value.endswith("%")


def test_changing_an_input_updates_the_prediction_live():
    app = open_page("Predict")
    before = app.metric[0].value
    app.selectbox(key="contract").select("Two year").run()
    assert app.metric[0].value != before


def test_predict_explains_the_individual_prediction():
    app = open_page("Predict")
    assert any("Why this score" in s.value for s in app.subheader)
    assert len(app.get("image")) >= 1


def test_total_charges_follow_tenure_and_monthly_charges():
    app = open_page("Predict")
    app.slider(key="tenure").set_value(24).run()
    app.slider(key="monthly").set_value(50.0).run()
    assert app.number_input(key="total").value == pytest.approx(1200.0)


def test_add_ons_are_disabled_without_internet_service():
    app = open_page("Predict")
    assert not app.checkbox(key="TechSupport").disabled
    app.selectbox(key="internet").select("No").run()
    assert app.checkbox(key="TechSupport").disabled


def test_multiple_lines_is_disabled_without_phone_service():
    app = open_page("Predict")
    assert not app.selectbox(key="multiple_lines").disabled
    app.radio(key="phone").set_value("No").run()
    assert app.selectbox(key="multiple_lines").disabled


def test_top_risk_table_uses_original_units():
    table = open_page("Predict").dataframe[0].value
    assert len(table) == 10
    assert "customerID" in table.columns
    assert table["tenure"].ge(0).all()
    assert table["MonthlyCharges"].ge(18).all()
    assert table["Churn probability"].is_monotonic_decreasing


def test_rendering_the_app_never_rewrites_artifacts():
    st.cache_data.clear()
    before = {p.name: p.stat().st_mtime_ns for p in MODELS.iterdir()}
    open_page("Predict")
    assert {p.name: p.stat().st_mtime_ns for p in MODELS.iterdir()} == before
