"""
===========================================================
TradeVision AI

Train XGBoost Model
===========================================================
"""

from xgboost import XGBClassifier

from src.models.evaluate import evaluate_model
from src.models.save_model import save_model

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)


def train_xgboost_model():

    X, y = prepare_dataset("GC=F")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(
        X_train,
        y_train,
    )

    print()

    print("=" * 60)
    print("XGBOOST")
    print("=" * 60)

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    save_model(
        model,
        "xgboost.pkl",
    )

    return model


if __name__ == "__main__":

    train_xgboost_model()