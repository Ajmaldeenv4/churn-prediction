"""The persisted artifacts must agree with each other and with what the app serves."""
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import preprocess as pp

MODELS = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture(scope="module")
def comparison() -> pd.DataFrame:
    return pd.read_csv(MODELS / "comparison.csv").set_index("model")


@pytest.fixture(scope="module")
def holdout() -> pd.DataFrame:
    return pd.read_csv(MODELS / "holdout_predictions.csv")


@pytest.mark.parametrize("model", ["LogisticRegression", "RandomForest", "XGBoost"])
def test_reported_cv_score_agrees_with_the_test_score(comparison, model):
    # A large gap means something is leaking into the CV folds.
    row = comparison.loc[model]
    assert abs(row["cv_roc_auc_mean"] - row["roc_auc"]) < 0.03


def test_holdout_table_is_in_original_units(holdout):
    assert len(holdout) == 1409
    assert holdout["customerID"].is_unique
    assert holdout["tenure"].between(0, 72).all()
    assert holdout["MonthlyCharges"].between(18, 120).all()
    assert holdout["churn_probability"].between(0, 1).all()


def test_holdout_table_reproduces_the_reported_auc(holdout, comparison):
    auc = roc_auc_score(holdout["actual_churn"], holdout["churn_probability"])
    assert auc == pytest.approx(comparison.loc["XGBoost", "roc_auc"], abs=0.001)


def test_serving_path_reproduces_the_recorded_probabilities(holdout):
    with open(MODELS / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    scaler = joblib.load(MODELS / "scaler.pkl")
    names = json.loads((MODELS / "feature_names.json").read_text())
    raw = pp.load_raw().set_index("customerID")

    for _, recorded in holdout.head(25).iterrows():
        customer = raw.loc[recorded["customerID"]].drop("Churn").to_dict()
        features = scaler.transform(pp.encode_customer(customer, names))
        served = float(model.predict_proba(features)[0, 1])
        assert served == pytest.approx(recorded["churn_probability"], abs=1e-3)


def test_smote_comparison_covers_both_strategies():
    smote = pd.read_csv(MODELS / "smote_comparison.csv").set_index("strategy")
    assert list(smote.index) == ["No resampling", "SMOTE in each fold"]
    assert {"cv_roc_auc", "cv_brier", "cv_recall", "mean_predicted"} <= set(smote.columns)


def test_shap_values_cover_every_holdout_row_and_feature(holdout):
    names = json.loads((MODELS / "feature_names.json").read_text())
    assert np.load(MODELS / "shap_values.npy").shape == (len(holdout), len(names))
