"""
===========================================================
TradeVision AI

Signal Performance Analysis
===========================================================
"""

import pandas as pd

from src.config.settings import PROCESSED_DATA_DIR


def analyze_signals():

    results_file = (
        PROCESSED_DATA_DIR / "backtest_results.csv"
    )

    data = pd.read_csv(
        results_file,
        index_col=0,
        parse_dates=True,
    )

    bullish = data[
        data["Prediction"] == 1
    ]["Next_Day_Return"]

    no_signal = data[
        data["Prediction"] == 0
    ]["Next_Day_Return"]

    print()

    print("=" * 60)
    print("SIGNAL PERFORMANCE ANALYSIS")
    print("=" * 60)

    print()

    print(
        f"Bullish Signal Days: "
        f"{len(bullish)}"
    )

    print(
        f"No-Signal Days: "
        f"{len(no_signal)}"
    )

    print()

    print(
        f"Average Return After Bullish Signal: "
        f"{bullish.mean():.4%}"
    )

    print(
        f"Average Return After No Signal: "
        f"{no_signal.mean():.4%}"
    )

    print()

    print(
        f"Median Return After Bullish Signal: "
        f"{bullish.median():.4%}"
    )

    print(
        f"Median Return After No Signal: "
        f"{no_signal.median():.4%}"
    )

    print()

    print(
        f"Bullish Signal Win Rate: "
        f"{(bullish > 0).mean():.2%}"
    )

    print(
        f"No-Signal Positive Rate: "
        f"{(no_signal > 0).mean():.2%}"
    )


if __name__ == "__main__":

    analyze_signals()