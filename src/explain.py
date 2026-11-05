"""Generate SHAP explanations for the trained XGBoost model.

Outputs in app/assets:
  - shap_summary.png            beeswarm summary (global)
  - shap_bar.png                mean |SHAP| ranked
  - shap_waterfall_churner.png  one true-positive walk-through
  - shap_waterfall_nonchurner.png  contrast: non-churner
  - shap_dependence_monthly.png  MonthlyCharges dependence

Also writes raw values to models/shap_values.npy.

The model consumes scaled features, but every plot labels them in original units
(tenure in months, charges in dollars) so the charts can be read directly.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from preprocess import split_data

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ASSETS = ROOT / "app" / "assets"


def main() -> None:
    _, X_original, _, y_test, feature_names, _ = split_data()
    X_display = pd.DataFrame(X_original, columns=feature_names)
    X_test = pd.DataFrame(
        joblib.load(MODELS_DIR / "scaler.pkl").transform(X_original), columns=feature_names
    )

    with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Building TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    scored = explainer(X_test)
    shap_values = scored.values
    explanation = shap.Explanation(
        values=shap_values, base_values=scored.base_values,
        data=X_original, feature_names=feature_names,
    )

    np.save(MODELS_DIR / "shap_values.npy", shap_values)
    print(f"Saved shap_values.npy  shape={shap_values.shape}")

    # 1. Beeswarm summary
    shap.summary_plot(shap_values, X_display, show=False)
    plt.savefig(ASSETS / "shap_summary.png", bbox_inches="tight", dpi=150)
    plt.clf()

    # 2. Bar chart (mean |SHAP|)
    shap.summary_plot(shap_values, X_display, plot_type="bar", show=False)
    plt.savefig(ASSETS / "shap_bar.png", bbox_inches="tight", dpi=150)
    plt.clf()

    # Pick a high-probability churner and a confident non-churner from test set.
    proba = model.predict_proba(X_test)[:, 1]
    churner_idx = int(np.argmax(proba * (y_test == 1)))
    nonchurner_idx = int(np.argmin(proba + (y_test == 1) * 10))

    print(f"churner idx={churner_idx}   p(churn)={proba[churner_idx]:.3f}   actual={y_test[churner_idx]}")
    print(f"non-churner idx={nonchurner_idx}   p(churn)={proba[nonchurner_idx]:.3f}   actual={y_test[nonchurner_idx]}")

    # 3. Waterfall — churner
    shap.plots.waterfall(explanation[churner_idx], show=False, max_display=12)
    plt.savefig(ASSETS / "shap_waterfall_churner.png", bbox_inches="tight", dpi=150)
    plt.clf()

    # 4. Waterfall — non-churner
    shap.plots.waterfall(explanation[nonchurner_idx], show=False, max_display=12)
    plt.savefig(ASSETS / "shap_waterfall_nonchurner.png", bbox_inches="tight", dpi=150)
    plt.clf()

    # 5. Dependence plot — MonthlyCharges
    shap.dependence_plot("MonthlyCharges", shap_values, X_display, show=False)
    plt.savefig(ASSETS / "shap_dependence_monthly.png", bbox_inches="tight", dpi=150)
    plt.clf()

    print("All SHAP plots saved to", ASSETS)


if __name__ == "__main__":
    main()
