"""
===========================================================
TradeVision AI

Prediction Report
===========================================================
"""

from src.models.load_model import load_model
from src.models.prepare_data import prepare_dataset


def generate_report():

    model = load_model(
        "tradevision_final.pkl"
    )

    X, y = prepare_dataset("GC=F")

    latest_features = X.iloc[[-1]]

    latest_date = latest_features.index[0]

    prediction = model.predict(
        latest_features
    )[0]

    probability = model.predict_proba(
        latest_features
    )[0][1]

    print()

    print("=" * 60)
    print("TRADEVISION AI — MARKET PREDICTION REPORT")
    print("=" * 60)

    print()

    print(
        f"Date: {latest_date.date()}"
    )

    print()

    print(
        f"Closing Price: "
        f"{latest_features['Close'].iloc[0]:.2f}"
    )

    print(
        f"RSI: "
        f"{latest_features['RSI'].iloc[0]:.2f}"
    )

    print(
        f"MACD: "
        f"{latest_features['MACD'].iloc[0]:.4f}"
    )

    print(
        f"Volatility: "
        f"{latest_features['Volatility'].iloc[0]:.4f}"
    )

    print()

    print(
        f"Probability of Upward Movement: "
        f"{probability:.2%}"
    )

    print()

    if prediction == 1:

        print(
            "Model Signal: BULLISH"
        )

    else:

        print(
            "Model Signal: NOT BULLISH"
        )

    print()

    print("=" * 60)


if __name__ == "__main__":

    generate_report()