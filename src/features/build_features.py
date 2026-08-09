"""
===========================================================
TradeVision AI

Feature Engineering

Purpose:
Create machine learning features from
historical financial market data.
===========================================================
"""

import numpy as np
import pandas as pd

from src.data.load_data import load_market_data
from src.config.settings import PROCESSED_DATA_DIR


def add_daily_return(data: pd.DataFrame) -> pd.DataFrame:
    """
    Percentage daily return.
    """

    data = data.copy()

    data["Daily_Return"] = data["Close"].pct_change()

    return data


def add_log_return(data: pd.DataFrame) -> pd.DataFrame:
    """
    Logarithmic return.
    """

    data = data.copy()

    data["Log_Return"] = np.log(
        data["Close"] / data["Close"].shift(1)
    )

    return data


def add_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    """
    Simple Moving Averages.
    """

    data = data.copy()

    data["SMA_20"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    data["SMA_50"] = (
        data["Close"]
        .rolling(50)
        .mean()
    )

    data["SMA_200"] = (
        data["Close"]
        .rolling(200)
        .mean()
    )

    return data


def add_rsi(
    data: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).
    """

    data = data.copy()

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()

    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (100 / (1 + rs))

    return data


def add_ema(
    data: pd.DataFrame,
    period: int = 20,
) -> pd.DataFrame:
    """
    Add Exponential Moving Average.
    """

    data = data.copy()

    data[f"EMA_{period}"] = (
        data["Close"]
        .ewm(span=period, adjust=False)
        .mean()
    )

    return data


def add_macd(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add MACD and Signal Line.
    """

    data = data.copy()

    ema12 = data["Close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    data["MACD"] = ema12 - ema26

    data["MACD_Signal"] = (
        data["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    return data


def add_volatility(
    data: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Add rolling volatility.
    """

    data = data.copy()

    data["Volatility"] = (
        data["Daily_Return"]
        .rolling(window)
        .std()
    )

    return data


def add_bollinger_bands(
    data: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Add Bollinger Bands.
    """

    data = data.copy()

    rolling_mean = (
        data["Close"]
        .rolling(window)
        .mean()
    )

    rolling_std = (
        data["Close"]
        .rolling(window)
        .std()
    )

    data["BB_Middle"] = rolling_mean

    data["BB_Upper"] = (
        rolling_mean + (2 * rolling_std)
    )

    data["BB_Lower"] = (
        rolling_mean - (2 * rolling_std)
    )

    return data


def add_atr(
    data: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Add Average True Range.
    """

    data = data.copy()

    high_low = data["High"] - data["Low"]

    high_close = (
        data["High"] -
        data["Close"].shift()
    ).abs()

    low_close = (
        data["Low"] -
        data["Close"].shift()
    ).abs()

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1,
    ).max(axis=1)

    data["ATR"] = (
        true_range
        .rolling(period)
        .mean()
    )

    return data


def add_time_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add calendar features.
    """

    data = data.copy()

    data["DayOfWeek"] = data.index.dayofweek

    data["Month"] = data.index.month

    data["Quarter"] = data.index.quarter

    return data


def add_target(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create prediction target.

    1 = Tomorrow's close is higher.
    0 = Tomorrow's close is lower or equal.
    """

    data = data.copy()

    data["Target"] = (
        data["Close"]
        .shift(-1)
        > data["Close"]
    ).astype(int)

    return data


def save_feature_dataset(
    data: pd.DataFrame,
    symbol: str,
) -> None:
    """
    Save engineered feature dataset.
    """

    filename = f"{symbol.replace('=', '_')}_features.csv"

    save_path = PROCESSED_DATA_DIR / filename

    data.to_csv(save_path)

    print("\nFeature dataset saved successfully.")

    print(save_path)


def build_feature_set(
    data: pd.DataFrame,
) -> pd.DataFrame:

    data = add_daily_return(data)

    data = add_log_return(data)

    data = add_moving_averages(data)

    data = add_rsi(data)

    data = add_ema(data)

    data = add_macd(data)

    data = add_volatility(data)

    data = add_bollinger_bands(data)

    data = add_atr(data)

    data = add_time_features(data)

    data = add_target(data)

    return data


if __name__ == "__main__":

    df = load_market_data("GC=F")

    df = build_feature_set(df)

save_feature_dataset(df, "GC=F")

print(
    df[
        [
            "Close",
            "EMA_20",
            "MACD",
            "RSI",
            "ATR",
            "BB_Upper",
            "BB_Lower",
            "DayOfWeek",
            "Month",
            "Target",
        ]
    ].tail(20)
)