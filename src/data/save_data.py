"""
===========================================================
TradeVision AI

Save Data Module

Purpose:
Save validated datasets to disk.
===========================================================
"""

import pandas as pd

from pathlib import Path


def save_market_data(
    data: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Save a dataset to CSV.

    Parameters
    ----------
    data : pandas.DataFrame

    save_path : Path
    """

    data.to_csv(save_path)

    print(f"\nDataset saved successfully")

    print(save_path)