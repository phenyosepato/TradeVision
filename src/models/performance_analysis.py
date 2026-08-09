"""
===========================================================
TradeVision AI

Strategy Performance Analysis
===========================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.config.settings import PROCESSED_DATA_DIR


def calculate_max_drawdown(equity_curve):

    running_max = equity_curve.cummax()

    drawdown = (
        equity_curve / running_max
    ) - 1

    return drawdown.min()


def analyze_performance():

    # -----------------------------------------------------
    # LOAD BACKTEST RESULTS
    # -----------------------------------------------------

    results_file = (
        PROCESSED_DATA_DIR / "backtest_results.csv"
    )

    data = pd.read_csv(
        results_file,
        index_col=0,
        parse_dates=True,
    )

    # -----------------------------------------------------
    # PERFORMANCE METRICS
    # -----------------------------------------------------

    strategy_equity = data[
        "Strategy_Equity"
    ]

    buy_hold_equity = data[
        "Buy_Hold_Equity"
    ]

    strategy_returns = data[
        "Strategy_Return"
    ].dropna()

    # -----------------------------------------------------
    # TOTAL RETURNS
    # -----------------------------------------------------

    strategy_total_return = (
        strategy_equity.iloc[-1] - 1
    )

    buy_hold_total_return = (
        buy_hold_equity.iloc[-1] - 1
    )

    # -----------------------------------------------------
    # ANNUALIZED RETURNS
    # -----------------------------------------------------

    number_of_years = (
        len(data) / 252
    )

    strategy_annual_return = (
        (1 + strategy_total_return)
        ** (1 / number_of_years)
    ) - 1

    buy_hold_annual_return = (
        (1 + buy_hold_total_return)
        ** (1 / number_of_years)
    ) - 1

    # -----------------------------------------------------
    # VOLATILITY
    # -----------------------------------------------------

    annualized_volatility = (
        strategy_returns.std()
        * (252 ** 0.5)
    )

    # -----------------------------------------------------
    # SHARPE RATIO
    # -----------------------------------------------------

    if annualized_volatility != 0:

        sharpe_ratio = (
            strategy_returns.mean()
            / strategy_returns.std()
        ) * (252 ** 0.5)

    else:

        sharpe_ratio = 0

    # -----------------------------------------------------
    # MAXIMUM DRAWDOWN
    # -----------------------------------------------------

    strategy_max_drawdown = (
        calculate_max_drawdown(
            strategy_equity
        )
    )

    buy_hold_max_drawdown = (
        calculate_max_drawdown(
            buy_hold_equity
        )
    )

    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    print()

    print("=" * 60)
    print("STRATEGY PERFORMANCE ANALYSIS")
    print("=" * 60)

    print()

    print(
        f"Strategy Total Return: "
        f"{strategy_total_return:.2%}"
    )

    print(
        f"Buy & Hold Total Return: "
        f"{buy_hold_total_return:.2%}"
    )

    print()

    print(
        f"Strategy Annualized Return: "
        f"{strategy_annual_return:.2%}"
    )

    print(
        f"Buy & Hold Annualized Return: "
        f"{buy_hold_annual_return:.2%}"
    )

    print()

    print(
        f"Annualized Volatility: "
        f"{annualized_volatility:.2%}"
    )

    print(
        f"Sharpe Ratio: "
        f"{sharpe_ratio:.4f}"
    )

    print()

    print(
        f"Strategy Maximum Drawdown: "
        f"{strategy_max_drawdown:.2%}"
    )

    print(
        f"Buy & Hold Maximum Drawdown: "
        f"{buy_hold_max_drawdown:.2%}"
    )

    # -----------------------------------------------------
    # DRAW PERFORMANCE GRAPH
    # -----------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        data.index,
        strategy_equity,
        label="Model Strategy",
    )

    plt.plot(
        data.index,
        buy_hold_equity,
        label="Buy & Hold",
    )

    plt.title(
        "TradeVision AI — Strategy Performance"
    )

    plt.xlabel("Date")

    plt.ylabel(
        "Growth of $1"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    analyze_performance()