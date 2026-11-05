"""Data-handling rules: clean inputs, no leakage, and no training/serving skew."""
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE

import preprocess as pp

TEST_ROWS = 1409


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pp.load_raw()


@pytest.fixture(scope="module")
def encoded(raw) -> pd.DataFrame:
    return pp.encode(raw).drop(columns=["Churn"])


def customer(raw: pd.DataFrame, position: int = 0, **overrides) -> dict:
    row = raw.iloc[position].drop(["customerID", "Churn"]).to_dict()
    return {**row, **overrides}


def test_brand_new_customers_have_zero_total_charges(raw):
    # The 11 blank TotalCharges are all tenure-0 customers: nothing billed yet.
    brand_new = raw[raw["tenure"] == 0]
    assert len(brand_new) == 11
    assert (brand_new["TotalCharges"] == 0).all()


def test_split_data_returns_held_out_rows_in_original_units(raw):
    _, X_test, _, y_test, names, test_index = pp.split_data()
    assert len(X_test) == TEST_ROWS
    tenure = X_test[:, names.index("tenure")]
    np.testing.assert_array_equal(tenure, raw.loc[test_index, "tenure"].values)
    np.testing.assert_array_equal(y_test, (raw.loc[test_index, "Churn"] == "Yes").astype(int))


def test_split_data_writes_no_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "MODELS_DIR", tmp_path)
    pp.split_data()
    assert list(tmp_path.iterdir()) == []


def test_prepare_data_resamples_only_the_training_fold(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "MODELS_DIR", tmp_path)
    _, X_test, y_train, y_test, _ = pp.prepare_data()
    assert y_train.mean() == 0.5
    assert len(X_test) == TEST_ROWS
    assert y_test.mean() == pytest.approx(0.265, abs=0.005)


def test_scaler_is_fit_on_the_training_fold_only(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "MODELS_DIR", tmp_path)
    X_train, X_test, *_ = pp.prepare_data()
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-9)
    assert not np.allclose(X_test.mean(axis=0), 0, atol=1e-3)


def test_encode_customer_matches_the_training_encoding(raw, encoded):
    names = encoded.columns.tolist()
    for position in range(0, len(raw), 50):
        row = pp.encode_customer(customer(raw, position), names)
        np.testing.assert_array_equal(row[0], encoded.iloc[position].values.astype(float))


def test_no_internet_service_clears_every_add_on(raw, encoded):
    names = encoded.columns.tolist()
    row = pp.encode_customer(
        customer(raw, InternetService="No", OnlineSecurity="Yes", TechSupport="Yes"), names
    )[0]
    assert row[names.index("OnlineSecurity")] == 0
    assert row[names.index("TechSupport")] == 0
    assert row[names.index("InternetService_No")] == 1


def test_no_phone_service_forces_the_matching_lines_category(raw, encoded):
    names = encoded.columns.tolist()
    row = pp.encode_customer(customer(raw, PhoneService="No", MultipleLines="Yes"), names)[0]
    assert row[names.index("MultipleLines_No phone service")] == 1
    assert row[names.index("MultipleLines_Yes")] == 0


def test_plain_smote_leaves_fractional_values_in_most_synthetic_rows():
    # Backs the claim on the Model Comparison page and in the README.
    X_train, _, y_train, *_ = pp.split_data()
    resampled, _ = SMOTE(random_state=42).fit_resample(X_train, y_train)
    synthetic = resampled[len(X_train):]
    names = pp.encode(pp.load_raw()).drop(columns=["Churn"]).columns
    dummies = [i for i, name in enumerate(names)
               if name not in ("tenure", "MonthlyCharges", "TotalCharges")]
    fractional = ((synthetic[:, dummies] > 0) & (synthetic[:, dummies] < 1)).any(axis=1)
    assert fractional.mean() > 0.8
