"""
===========================================================
TradeVision AI

Prepare Machine Learning Dataset
===========================================================
"""

import pandas as pd

from src.data.load_data import load_feature_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_dataset(symbol: str):

    data = load_feature_data(symbol)

    # Remove rows containing missing values
    data = data.dropna()

    # Separate target
    X = data.drop(columns=["Target"])

    y = data["Target"]

    return X, y


def split_dataset(X, y, test_size=0.2):
    """
    Split dataset and scale features.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X, y = prepare_dataset("GC=F")

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    print()

    print(f"Training features: {X_train.shape}")
    print(f"Testing features : {X_test.shape}")

    print()

    print(f"Training target  : {y_train.shape}")
    print(f"Testing target   : {y_test.shape}")