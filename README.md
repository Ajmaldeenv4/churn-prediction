# Telco Customer Churn — End-to-End ML

Predict whether a Telco customer will churn, explain every prediction with SHAP, and serve the whole thing through a 4-page Streamlit dashboard.

**🔗 Live demo:** https://ajmaldeenv4-churn-prediction-appapp-5klujp.streamlit.app/
**📦 Repo:** https://github.com/Ajmaldeenv4/churn-prediction

## Results

| Model | Test Accuracy | Test ROC-AUC | Test F1 | 5-fold CV ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.730 | 0.839 | 0.606 | 0.845 |
| Random Forest | 0.783 | 0.826 | 0.568 | 0.821 |
| **XGBoost (tuned)** | **0.795** | **0.846** | **0.612** | **0.847** |

Tuned XGBoost via `GridSearchCV` on `n_estimators`, `max_depth`, `learning_rate` with `scoring='roc_auc'`, refitting SMOTE and scaling inside every fold. Best params: `{learning_rate: 0.05, max_depth: 3, n_estimators: 200}`.

CV and test scores agree to within about 0.01 for every model, which is the sign that nothing is leaking into the folds. XGBoost's lead over Logistic Regression is small (CV 0.847 vs. 0.845). It is served because it captures non-linear effects and SHAP's TreeExplainer gives exact per-customer explanations for it.

### Does SMOTE help?

`train.py` also scores the tuned XGBoost with and without SMOTE on identical folds (`models/smote_comparison.csv`):

| Strategy | CV ROC-AUC | CV Brier | CV recall at 0.5 | Mean predicted (actual 0.265) |
|---|---|---|---|---|
| No resampling | 0.847 | 0.134 | 0.526 | 0.266 |
| SMOTE in each fold | 0.846 | 0.136 | 0.599 | 0.290 |

SMOTE raises recall at the 0.5 cut-off but does not improve ranking, and its probabilities are slightly worse. Plain SMOTE interpolates one-hot columns, so most synthetic rows contain fractional values no real customer has, and trees can isolate those rows instead of learning general churn patterns.

## Dataset

- **Telco Customer Churn** (IBM sample): 7043 rows × 21 cols (19 input features, plus `customerID` and the target)
- Target: `Churn` (Yes/No → 1/0)
- Imbalance: ~26.5% churn

## Pipeline

1. **EDA** (`notebooks/01_eda.ipynb`) — donut, categorical churn-rate bars, numeric distributions, correlation heatmap.
2. **Preprocess** (`src/preprocess.py`) — drop `customerID`, coerce `TotalCharges` (the 11 blanks are unbilled tenure-0 customers, so they become 0), binary encode Yes/No, one-hot encode `Contract`/`InternetService`/`PaymentMethod`/`MultipleLines`, 80/20 stratified split, **SMOTE on train only**, **StandardScaler fit on train only**. Persists `models/scaler.pkl` and `models/feature_names.json`.
3. **Train** (`src/train.py`) — LR, RF, XGBoost with 5-fold CV + grid search, with SMOTE and scaling inside each fold. Writes `models/comparison.csv`, `models/smote_comparison.csv`, `models/holdout_predictions.csv`, `models/xgb_model.pkl`, `app/assets/roc_curves.png`, `app/assets/confusion_matrix.png`.
4. **Explain** (`src/explain.py`) — SHAP TreeExplainer → 5 plots in `app/assets/`, labelled in original units, + `models/shap_values.npy`.
5. **Dashboard** (`app/app.py`) — Overview / Model Comparison / Predict / Explain pages. Predict scores a customer live and shows a SHAP waterfall for that prediction.

## Setup

```bash
conda create -n churn python=3.10 -y
conda activate churn
pip install -r requirements.txt
```

The dataset, trained model, and plots are committed, so the dashboard runs right after install. Use `requirements-dev.txt` instead if you also want the tests and the notebook.

## Run

```bash
# in repo root, churn env active
streamlit run app/app.py
# http://localhost:8501
```

To rebuild every artifact from the CSV:

```bash
python src/preprocess.py
python src/train.py
python src/explain.py
```

## Test

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

The tests cover the leakage rules, training/serving parity, artifact consistency, and every dashboard page (driven headlessly with Streamlit's `AppTest`). GitHub Actions runs them on every push to `main` and on pull requests.

## Folder layout

```
churn-prediction/
├── .github/workflows/
│   └── ci.yml
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
│   ├── smote_comparison.csv
│   ├── holdout_predictions.csv
│   └── shap_values.npy
├── app/
│   ├── app.py
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
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Key correctness rules (enforced in code and tests)

1. SMOTE is fit **only** on training data, including inside each CV fold. Synthetic rows never reach a validation fold or the test set.
2. StandardScaler is fit **only** on the training fold; the test fold is `.transform()`-ed.
3. Train/test split uses `stratify=y` to preserve the 26/74 ratio in the held-out set.
4. All metrics are reported on the original (non-SMOTE) test fold.
5. Every artifact (model, scaler, feature names, plots, SHAP values, held-out scores) is persisted — Streamlit reads from disk only and never rewrites them.
6. Training and the dashboard share one encoder, so a customer is encoded identically in both.

## Notes

- `requirements.txt` is pinned because `models/*.pkl` must be loaded by the scikit-learn and XGBoost versions that wrote them. If you upgrade either, rerun the pipeline.
- `xgboost` stays below 3.0 because SHAP 0.49 cannot parse the XGBoost 3.x `base_score` serialization format.
- The Predict page builds the feature vector through `encode_customer()` and the persisted `feature_names.json` to guarantee column order matches what the model was trained on.

## Interview talking points

- **The leak I found:** applying SMOTE before cross-validation scored about 0.94 ROC-AUC against 0.84 on the test set, because neighbouring synthetic rows landed on both sides of each split. Moving SMOTE inside the folds brought CV to 0.847, and the honest search chose a different learning rate.
- **Why SMOTE?** It was the first thing to try for a 26/74 split, but the measurements show it only trades a little probability quality for recall at the 0.5 cut-off. Without resampling, XGBoost still ranks customers just as well (see *Does SMOTE help?*), so this comparison is worth raising rather than assuming SMOTE helps.
- **Why XGBoost?** Mixed feature types, robust to outliers, built-in L1/L2 regularization, fast on tabular data. Best CV and test ROC-AUC here, though only narrowly ahead of Logistic Regression.
- **Why SHAP over native feature importance?** Native importance only tells you *which* features the model splits on. SHAP gives a per-prediction, signed, magnitude-aware decomposition grounded in Shapley values from cooperative game theory.
- **Reading a waterfall plot:** every bar shows how a single feature moved that prediction above or below the model's baseline. Red = pushed toward churn, blue = pushed away.
