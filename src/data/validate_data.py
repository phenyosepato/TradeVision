"""
===========================================================
TradeVision AI

Data Validation

Purpose:
Validate downloaded market datasets before they
enter the machine learning pipeline.
===========================================================
"""

import pandas as pd


REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def validate_market_data(data: pd.DataFrame) -> bool:
    """
    Validate a financial dataset.

    Returns
    -------
    bool
        True if dataset passes validation.
    """

    print("\n" + "=" * 60)
    print("VALIDATING DATASET")
    print("=" * 60)

    # -----------------------------
    # Empty Dataset
    # -----------------------------
    if data.empty:
        raise ValueError("Dataset is empty.")

    print("✓ Dataset is not empty")

    # -----------------------------
    # Flatten MultiIndex Columns
    # -----------------------------
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # -----------------------------
    # Required Columns
    # -----------------------------
    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("✓ Required columns exist")

    # -----------------------------
    # Duplicate Dates
    # -----------------------------
    duplicates = data.index.duplicated().sum()

    if duplicates > 0:
        raise ValueError(
            f"{duplicates} duplicate dates found."
        )

    print("✓ No duplicate dates")

    # -----------------------------
    # Missing Values
    # -----------------------------
    total_missing = data.isna().sum().sum()

    if total_missing > 0:
        raise ValueError(
            f"{total_missing} missing values found."
        )

    print("✓ No missing values")

    # -----------------------------
    # Negative Prices
    # -----------------------------
    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for column in price_columns:

        if (data[column] < 0).any():
            raise ValueError(
                f"Negative prices found in {column}"
            )

    print("✓ Prices are valid")

    # -----------------------------
    # Date Order
    # -----------------------------
    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "Dates are not sorted."
        )

    print("✓ Dates are sorted")

    # -----------------------------
    # Dataset Summary
    # -----------------------------
    print("\nDataset Summary")

    print("-" * 30)

    print(f"Rows: {len(data):,}")

    print(f"Columns: {len(data.columns)}")

    print(
        f"Start: {data.index.min().date()}"
    )

    print(
        f"End: {data.index.max().date()}"
    )

    print("\nValidation Successful\n")

    return True