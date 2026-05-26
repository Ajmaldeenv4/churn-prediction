# Telco Customer Churn — End-to-End ML

Predict whether a Telco customer will churn, explain every prediction with SHAP, and serve the whole thing through a 4-page Streamlit dashboard.

## Results

| Model | Test Accuracy | Test ROC-AUC | Test F1 | 5-fold CV ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.730 | 0.839 | 0.605 | 0.849 |
| Random Forest | 0.780 | 0.825 | 0.557 | 0.938 |
| **XGBoost (tuned)** | **0.798** | **0.844** | 0.600 | **0.945** |

Tuned XGBoost via `GridSearchCV` on `n_estimators`, `max_depth`, `learning_rate` with `scoring='roc_auc'`. Best params: `{learning_rate: 0.1, max_depth: 3, n_estimators: 200}`.

## Dataset

- **Telco Customer Churn** (IBM sample): 7043 rows × 21 cols
- Target: `Churn` (Yes/No → 1/0)
- Imbalance: ~26.5% churn

## Pipeline

1. **EDA** (`notebooks/01_eda.ipynb`) — donut, categorical churn-rate bars, numeric distributions, correlation heatmap.
2. **Preprocess** (`src/preprocess.py`) — drop `customerID`, coerce `TotalCharges`, binary encode Yes/No, one-hot encode `Contract`/`InternetService`/`PaymentMethod`/`MultipleLines`, 80/20 stratified split, **SMOTE on train only**, **StandardScaler fit on train only**. Persists `models/scaler.pkl` and `models/feature_names.json`.
3. **Train** (`src/train.py`) — LR, RF, XGBoost with 5-fold CV + grid search. Writes `models/comparison.csv`, `models/xgb_model.pkl`, `app/assets/roc_curves.png`, `app/assets/confusion_matrix.png`.
4. **Explain** (`src/explain.py`) — SHAP TreeExplainer → 5 plots in `app/assets/` + `models/shap_values.npy`.
5. **Dashboard** (`app/app.py`) — Overview / Model Comparison / Predict / Explain pages.

## Setup

```bash
conda create -n churn python=3.10 -y
conda activate churn
pip install -r requirements.txt
```

Drop `WA_Fn-UseC_-Telco-Customer-Churn.csv` into `data/`.

## Run

```bash
# in repo root, churn env active
python src/preprocess.py
python src/train.py
python src/explain.py
streamlit run app/app.py
# http://localhost:8501
```

## Folder layout

```
churn_prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── preprocess.py     # prepare_data() returns (X_train_res, X_test, y_train_res, y_test, feature_names)
│   ├── train.py
│   └── explain.py
├── models/
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   ├── comparison.csv
│   └── shap_values.npy
├── app/
│   ├── app.py
│   ├── preprocess_loader.py
│   └── assets/
│       ├── churn_donut.png
│       ├── churn_by_category.png
│       ├── numeric_distributions.png
│       ├── correlation_heatmap.png
│       ├── roc_curves.png
│       ├── confusion_matrix.png
│       ├── shap_summary.png
│       ├── shap_bar.png
│       ├── shap_waterfall_churner.png
│       ├── shap_waterfall_nonchurner.png
│       └── shap_dependence_monthly.png
├── requirements.txt
└── README.md
```

## Key correctness rules (enforced in code)

1. SMOTE is fit **only** on the training fold.
2. StandardScaler is fit **only** on the training fold; the test fold is `.transform()`-ed.
3. Train/test split uses `stratify=y` to preserve the 26/74 ratio in the held-out set.
4. All metrics are reported on the original (non-SMOTE) test fold.
5. Every artifact (model, scaler, feature names, plots, SHAP values) is persisted — Streamlit reads from disk only.

## Notes

- `xgboost` is pinned to `<3.0` because SHAP 0.49 cannot parse the XGBoost 3.x `base_score` serialization format.
- The Predict page rebuilds the feature vector using the persisted `feature_names.json` to guarantee column order matches what the model was trained on.

## Interview talking points

- **Why SMOTE?** Without balancing, a model learns "always predict no churn" and still hits ~74% accuracy. SMOTE creates synthetic minority samples — but only on the training fold, so test-set metrics stay honest.
- **Why XGBoost?** Mixed feature types, robust to outliers, built-in L1/L2 regularization, fast on tabular data. Wins both CV and test ROC-AUC here.
- **Why SHAP over native feature importance?** Native importance only tells you *which* features the model splits on. SHAP gives a per-prediction, signed, magnitude-aware decomposition grounded in Shapley values from cooperative game theory.
- **Reading a waterfall plot:** every bar shows how a single feature moved that prediction above or below the model's baseline. Red = pushed toward churn, blue = pushed away.
