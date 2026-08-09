"""
===========================================================
TradeVision AI

Statistical Significance Test
===========================================================
"""

import pandas as pd

from scipy.stats import ttest_ind

from src.config.settings import PROCESSED_DATA_DIR


def run_statistical_test():

    results_file = (
        PROCESSED_DATA_DIR / "backtest_results.csv"
    )

    data = pd.read_csv(
        results_file,
        index_col=0,
        parse_dates=True,
    )

    bullish_returns = data[
        data["Prediction"] == 1
    ]["Next_Day_Return"].dropna()

    no_signal_returns = data[
        data["Prediction"] == 0
    ]["Next_Day_Return"].dropna()

    statistic, p_value = ttest_ind(
        bullish_returns,
        no_signal_returns,
        equal_var=False,
    )

    print()

    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("=" * 60)

    print()

    print(
        f"Bullish Signal Mean: "
        f"{bullish_returns.mean():.4%}"
    )

    print(
        f"No-Signal Mean: "
        f"{no_signal_returns.mean():.4%}"
    )

    print()

    print(
        f"Difference: "
        f"{bullish_returns.mean() - no_signal_returns.mean():.4%}"
    )

    print(
        f"T-statistic: "
        f"{statistic:.4f}"
    )

    print(
        f"P-value: "
        f"{p_value:.4f}"
    )

    print()

    if p_value < 0.05:

        print(
            "Result: Statistically significant "
            "at the 5% level."
        )

    else:

        print(
            "Result: Not statistically significant "
            "at the 5% level."
        )


if __name__ == "__main__":

    run_statistical_test()