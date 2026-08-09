"""
===========================================================
TradeVision AI

Historical Backtest
===========================================================

Purpose:
Evaluate the Logistic Regression model as a historical
paper-trading signal.

Important:
The model only uses information available at time t to
predict the direction of the next trading day.

No future information is used to create the prediction.
===========================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)

from src.config.settings import PROCESSED_DATA_DIR
TRANSACTION_COST = 0.001

def run_backtest():

    # -----------------------------------------------------
    # LOAD MODEL DATA
    # -----------------------------------------------------

    X, y = prepare_dataset("GC=F")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
    )

    test_dates = X.index[len(X_train):]

    # -----------------------------------------------------
    # TRAIN MODEL
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # GENERATE TEST-SET PREDICTIONS
    # -----------------------------------------------------

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    predictions = pd.Series(
        predictions,
        index=test_dates,
        name="Prediction",
    )

    # -----------------------------------------------------
    # LOAD PRICE DATA
    # -----------------------------------------------------

    feature_file = (
        PROCESSED_DATA_DIR / "GC_F_features.csv"
    )

    data = pd.read_csv(
        feature_file,
        index_col=0,
        parse_dates=True,
    )

    # -----------------------------------------------------
    # ALIGN TEST DATA WITH PRICE DATA
    # -----------------------------------------------------

    test_data = data.loc[test_dates].copy()

    test_data["Prediction"] = predictions

    # -----------------------------------------------------
    # CALCULATE NEXT-DAY RETURN
    # -----------------------------------------------------

    test_data["Next_Day_Return"] = (
        test_data["Close"]
        .pct_change()
        .shift(-1)
    )

    # -----------------------------------------------------
    # STRATEGY RETURN
    # -----------------------------------------------------

    # Prediction 1 = bullish signal
    # Prediction 0 = no long position

    test_data["Position_Change"] = (
        test_data["Prediction"]
        .diff()
        .abs()
        .fillna(0)
    )

    test_data["Strategy_Return"] = (
        test_data["Prediction"]
        * test_data["Next_Day_Return"]
        - (
            test_data["Position_Change"]
            * TRANSACTION_COST
        )
    )

    # Remove final row because there is no next-day return
    test_data = test_data.dropna(
        subset=[
            "Next_Day_Return",
            "Strategy_Return",
        ]
    )

    # -----------------------------------------------------
    # CUMULATIVE RETURNS
    # -----------------------------------------------------

    test_data["Strategy_Equity"] = (
        1 + test_data["Strategy_Return"]
    ).cumprod()

    test_data["Buy_Hold_Equity"] = (
        1 + test_data["Next_Day_Return"]
    ).cumprod()

    # -----------------------------------------------------
    # PERFORMANCE METRICS
    # -----------------------------------------------------

    strategy_return = (
        test_data["Strategy_Equity"].iloc[-1] - 1
    )

    buy_hold_return = (
        test_data["Buy_Hold_Equity"].iloc[-1] - 1
    )

    number_of_signals = (
        test_data["Prediction"] == 1
    ).sum()

    winning_signals = (
        test_data.loc[
            test_data["Prediction"] == 1,
            "Next_Day_Return",
        ]
        > 0
    ).sum()

    if number_of_signals > 0:

        win_rate = (
            winning_signals
            / number_of_signals
        )

    else:

        win_rate = 0

    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    print()

    print("=" * 60)
    print("HISTORICAL BACKTEST")
    print("=" * 60)

    print()

    print(
        f"Test Period: "
        f"{test_data.index.min().date()} "
        f"to "
        f"{test_data.index.max().date()}"
    )

    print()

    print(
        f"Strategy Return: "
        f"{strategy_return:.2%}"
    )

    print(
        f"Buy & Hold Return: "
        f"{buy_hold_return:.2%}"
    )

    print(
        f"Number of Bullish Signals: "
        f"{number_of_signals}"
    )

    print(
        f"Signal Win Rate: "
        f"{win_rate:.2%}"
    )

    results_file = (
        PROCESSED_DATA_DIR / "backtest_results.csv"
    )

    test_data.to_csv(
        results_file
    )

    print()

    print("Backtest results saved to:")
    print(results_file)
    # -----------------------------------------------------
    # PERFORMANCE GRAPH
    # -----------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        test_data.index,
        test_data["Strategy_Equity"],
        label="Model Strategy",
    )

    plt.plot(
        test_data.index,
        test_data["Buy_Hold_Equity"],
        label="Buy & Hold",
    )

    plt.title(
        "TradeVision AI — Historical Backtest"
    )

    plt.xlabel("Date")

    plt.ylabel(
        "Growth of $1"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    run_backtest()