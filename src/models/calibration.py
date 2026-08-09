"""
===========================================================
TradeVision AI

Probability Calibration Analysis
===========================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)


def analyze_calibration():

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

    fraction_positive, mean_predicted = calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="uniform",
    )

    print()

    print("=" * 60)
    print("PROBABILITY CALIBRATION")
    print("=" * 60)

    print()

    print("Mean Predicted Probability:")
    print(mean_predicted)

    print()

    print("Actual Positive Fraction:")
    print(fraction_positive)

    # -----------------------------------------------------
    # CALIBRATION PLOT
    # -----------------------------------------------------

    plt.figure(figsize=(8, 8))

    plt.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label="Logistic Regression",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration",
    )

    plt.title(
        "TradeVision AI — Probability Calibration"
    )

    plt.xlabel(
        "Mean Predicted Probability"
    )

    plt.ylabel(
        "Fraction of Positive Outcomes"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    analyze_calibration()