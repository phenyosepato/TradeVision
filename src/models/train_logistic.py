"""
===========================================================
TradeVision AI

Train Logistic Regression Baseline
===========================================================
"""

from sklearn.linear_model import LogisticRegression
from src.models.evaluate import evaluate_model
from src.models.save_model import save_model

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)


def train_logistic_model():

    X, y = prepare_dataset("GC=F")

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    print()

    print("=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    save_model(
    model,
    "logistic_regression.pkl",
    )
    
    return model


if __name__ == "__main__":

    train_logistic_model()