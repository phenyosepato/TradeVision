"""
===========================================================
TradeVision AI

Naive Baseline Model
===========================================================
"""

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)


def evaluate_baseline():

    X, y = prepare_dataset("GC=F")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
    )

    model = DummyClassifier(
        strategy="most_frequent"
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print()

    print("=" * 60)
    print("NAIVE BASELINE")
    print("=" * 60)

    print(
        f"Baseline Accuracy: {accuracy:.4f}"
    )


if __name__ == "__main__":

    evaluate_baseline()