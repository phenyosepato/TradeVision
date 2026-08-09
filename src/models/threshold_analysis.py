"""
===========================================================
TradeVision AI

Prediction Threshold Analysis
===========================================================
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)


def analyze_thresholds():

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
                "logistic_regression",
                LogisticRegression(
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

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
    ]

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = (
            predictions == y_test
        ).mean()

        bullish_signals = (
            predictions == 1
        ).sum()

        results.append(
            {
                "Threshold": threshold,
                "Accuracy": accuracy,
                "Bullish Signals": bullish_signals,
            }
        )

    results_df = pd.DataFrame(results)

    print()

    print("=" * 60)
    print("PREDICTION THRESHOLD ANALYSIS")
    print("=" * 60)

    print()

    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":

    analyze_thresholds()