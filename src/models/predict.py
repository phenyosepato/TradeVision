"""
===========================================================
TradeVision AI

Prediction Module
===========================================================
"""

import pandas as pd

from src.models.load_model import load_model
from src.models.prepare_data import prepare_dataset


def predict_latest():

    # Load the final trained model
    model = load_model(
        "tradevision_final.pkl"
    )

    # Load prepared feature data
    X, y = prepare_dataset("GC=F")

    # Get the latest observation
    latest_features = X.iloc[[-1]]

    latest_date = latest_features.index[0]

    # Generate prediction
    prediction = model.predict(
        latest_features
    )[0]

    # Generate probability
    probability = model.predict_proba(
        latest_features
    )[0][1]

    print()

    print("=" * 60)
    print("TRADEVISION AI — LATEST PREDICTION")
    print("=" * 60)

    print()

    print(
        f"Date: {latest_date.date()}"
    )

    print()

    if prediction == 1:

        print(
            "Prediction: BULLISH"
        )

    else:

        print(
            "Prediction: NOT BULLISH"
        )

    print()

    print(
        f"Probability of Upward Movement: "
        f"{probability:.2%}"
    )


if __name__ == "__main__":

    predict_latest()