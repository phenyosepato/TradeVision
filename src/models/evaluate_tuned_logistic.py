"""
===========================================================
TradeVision AI

Evaluate Tuned Logistic Regression
===========================================================
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluate import evaluate_model
from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)


def evaluate_tuned_logistic():

    X, y = prepare_dataset("GC=F")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
    )

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=0.001,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    print()

    print("=" * 60)
    print("TUNED LOGISTIC REGRESSION")
    print("=" * 60)

    evaluate_model(
        model,
        X_test,
        y_test,
    )


if __name__ == "__main__":

    evaluate_tuned_logistic()