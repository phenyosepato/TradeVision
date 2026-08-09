"""
===========================================================
TradeVision AI

Exploratory Data Analysis
===========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.load_data import load_feature_data


def dataset_summary(data):

    print("\n")
    print("=" * 60)

    print("DATASET SUMMARY")

    print("=" * 60)

    print(f"Rows: {len(data):,}")

    print(f"Columns: {len(data.columns)}")

    print("\nColumns:\n")

    for column in data.columns:

        print(column)

    print("\n")

    print(data.describe())

    print("=" * 60)


def plot_price(data):

    plt.figure(figsize=(14, 6))

    plt.plot(
        data.index,
        data["Close"],
        linewidth=1.5,
    )

    plt.title("Gold Closing Price")

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def plot_daily_returns(data):

    plt.figure(figsize=(14, 6))

    plt.plot(
        data.index,
        data["Daily_Return"],
        linewidth=0.8,
    )

    plt.title("Daily Returns")

    plt.xlabel("Date")

    plt.ylabel("Return")

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def plot_volume(data):

    plt.figure(figsize=(14, 6))

    plt.plot(
        data.index,
        data["Volume"],
        linewidth=1,
    )

    plt.title("Trading Volume")

    plt.xlabel("Date")

    plt.ylabel("Volume")

    plt.grid(True)

    plt.tight_layout()

    plt.show()



def plot_correlation_heatmap(data):

    numeric_data = data.select_dtypes(include=[np.number])

    correlation = numeric_data.corr()

    plt.figure(figsize=(14, 12))

    plt.imshow(correlation)

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90,
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns,
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    plt.show()


def plot_return_distribution(data):

    plt.figure(figsize=(10, 6))

    plt.hist(
        data["Daily_Return"].dropna(),
        bins=50,
    )

    plt.title("Daily Return Distribution")

    plt.xlabel("Daily Return")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.show()


def plot_return_boxplot(data):

    plt.figure(figsize=(8, 6))

    plt.boxplot(
        data["Daily_Return"].dropna(),
    )

    plt.title("Daily Return Boxplot")

    plt.ylabel("Return")

    plt.tight_layout()

    plt.show()    




if __name__ == "__main__":

    df = load_feature_data("GC=F")

    dataset_summary(df)

    plot_price(df)

    plot_daily_returns(df)

    plot_volume(df)

    plot_correlation_heatmap(df)

    plot_return_distribution(df)

    plot_return_boxplot(df)