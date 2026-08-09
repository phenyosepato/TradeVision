"""
===========================================================
TradeVision AI

Data Profiling Module

Purpose:
Generate summary statistics for downloaded datasets.
===========================================================
"""

import pandas as pd


def generate_data_profile(
    data: pd.DataFrame,
    asset_name: str,
) -> None:
    """
    Display a summary of the dataset.
    """

    # Flatten MultiIndex columns if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    print("\n" + "=" * 60)
    print("DATA PROFILE")
    print("=" * 60)

    print(f"Asset: {asset_name}")

    print(f"Rows: {len(data):,}")

    print(f"Columns: {len(data.columns)}")

    print(
        f"Date Range: {data.index.min().date()} → {data.index.max().date()}"
    )

    print(
        f"Missing Values: {data.isna().sum().sum()}"
    )

    print(
        f"Duplicate Rows: {data.duplicated().sum()}"
    )

    memory = (
        data.memory_usage(deep=True).sum() / 1024**2
    )

    print(f"Memory Usage: {memory:.2f} MB")

    print(
        f"Average Close Price: {data['Close'].mean():.2f}"
    )

    print(
        f"Highest Close Price: {data['Close'].max():.2f}"
    )

    print(
        f"Lowest Close Price: {data['Close'].min():.2f}"
    )

    print(
        f"Average Volume: {data['Volume'].mean():,.0f}"
    )

    print("=" * 60)