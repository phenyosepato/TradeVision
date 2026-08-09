"""
===========================================================
TradeVision AI

Trading Frequency Analysis
===========================================================
"""

import pandas as pd

from src.config.settings import PROCESSED_DATA_DIR


def analyze_trading_frequency():

    results_file = (
        PROCESSED_DATA_DIR / "backtest_results.csv"
    )

    data = pd.read_csv(
        results_file,
        index_col=0,
        parse_dates=True,
    )

    position_changes = (
        data["Prediction"]
        .diff()
        .abs()
        .fillna(0)
    )

    number_of_position_changes = int(
        position_changes.sum()
    )

    number_of_days = len(data)

    percentage_of_days_changed = (
        number_of_position_changes
        / number_of_days
    )

    print()

    print("=" * 60)
    print("TRADING FREQUENCY ANALYSIS")
    print("=" * 60)

    print()

    print(
        f"Test Days: "
        f"{number_of_days}"
    )

    print(
        f"Position Changes: "
        f"{number_of_position_changes}"
    )

    print(
        f"Position Change Rate: "
        f"{percentage_of_days_changed:.2%}"
    )

    print()

    print(
        f"Average Days Between Changes: "
        f"{number_of_days / max(number_of_position_changes, 1):.2f}"
    )


if __name__ == "__main__":

    analyze_trading_frequency()