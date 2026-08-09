"""
===========================================================
TradeVision AI

Load Data Module

Purpose:
Load previously downloaded datasets from the raw data folder.
===========================================================
"""

import pandas as pd

from src.config.settings import RAW_DATA_DIR


def load_market_data(symbol: str) -> pd.DataFrame:
    """
    Load a market dataset from disk.

    Parameters
    ----------
    symbol : str

    Returns
    -------
    pandas.DataFrame
    """

    filename = f"{symbol.replace('=', '_')}.csv"

    file_path = RAW_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}"
        )

    data = pd.read_csv(
        file_path,
        header=[0, 1],      # Handles yfinance MultiIndex columns
        index_col=0,
        parse_dates=True,
    )

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data

def load_feature_data(symbol: str) -> pd.DataFrame:
    """
    Load engineered feature dataset.
    """

    filename = f"{symbol.replace('=', '_')}_features.csv"

    file_path = (
        RAW_DATA_DIR.parent /
        "processed" /
        filename
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{file_path}"
        )

    data = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True,
    )

    return data