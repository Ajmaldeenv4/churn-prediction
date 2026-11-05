"""Cross-validation must evaluate SMOTE the way it will be used: inside each fold."""
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

import preprocess as pp
import train


def test_cv_pipeline_does_not_inflate_the_score():
    # Resampling before the folds scores about 0.94 on this data because copies
    # of the same synthetic rows land on both sides of the split. Resampling
    # inside each fold has to land near the honest held-out range instead.
    X_train, _, y_train, *_ = pp.split_data()
    classifier = XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", tree_method="hist", random_state=42,
    )
    score = cross_val_score(
        train.smote_pipeline(classifier), X_train, y_train,
        cv=train.CV, scoring="roc_auc",
    ).mean()
    assert 0.80 < score < 0.87
