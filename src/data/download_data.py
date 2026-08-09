"""
===========================================================
TradeVision AI

Market Data Downloader

Author: Phenyo

Purpose:
Downloads historical market data from Yahoo Finance
and saves it inside the project's raw data folder.

===========================================================
"""

import pandas as pd
import yfinance as yf

from src.config.settings import (
    RAW_DATA_DIR,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_INTERVAL,
)
from src.data.validate_data import validate_market_data
from src.data.save_data import save_market_data
from src.data.profile_data import generate_data_profile

def download_market_data(
    symbol: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    interval: str = DEFAULT_INTERVAL,
    save: bool = True,
) -> pd.DataFrame:
    """
    Download historical financial market data.

    Parameters
    ----------
    symbol : str
        Financial asset ticker.

    start_date : str
        Beginning of download.

    end_date : str
        End of download.

    interval : str
        Data interval.

    save : bool
        Save CSV file.

    Returns
    -------
    pandas.DataFrame
    """

    print("=" * 60)
    print(f"Downloading market data for {symbol}")
    print("=" * 60)

    try:

        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )

        if data.empty:
            raise ValueError("No data was returned.")

        print(f"Downloaded {len(data)} rows.")

        validate_market_data(data)

        generate_data_profile(data, symbol)

        if save:

            filename = f"{symbol.replace('=', '_')}.csv"

            save_path = RAW_DATA_DIR / filename

            save_market_data(data, save_path)

        return data

    except Exception as error:

        print("Download failed.")

        print(error)

        raise


if __name__ == "__main__":

    gold = download_market_data("GC=F")

    print()

    print(gold.head())