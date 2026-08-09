"""
===========================================================
TradeVision AI

Prediction Probability Analysis
===========================================================
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)

from src.config.settings import PROCESSED_DATA_DIR


def analyze_probabilities():

    X, y = prepare_dataset("GC=F")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
    )

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_dates = X.index[len(X_train):]

    results = pd.DataFrame(
        index=test_dates
    )

    results["Probability_Up"] = probabilities

    # -----------------------------------------------------
    # LOAD NEXT-DAY RETURNS
    # -----------------------------------------------------

    feature_file = (
        PROCESSED_DATA_DIR / "GC_F_features.csv"
    )

    price_data = pd.read_csv(
        feature_file,
        index_col=0,
        parse_dates=True,
    )

    results["Next_Day_Return"] = (
        price_data.loc[
            test_dates,
            "Close",
        ]
        .pct_change()
        .shift(-1)
    )

    results = results.dropna()

    # -----------------------------------------------------
    # CREATE CONFIDENCE GROUPS
    # -----------------------------------------------------

    results["Confidence_Group"] = pd.cut(
        results["Probability_Up"],
        bins=[
            0.0,
            0.5,
            0.6,
            0.7,
            0.8,
            1.0,
        ],
        labels=[
            "<50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80%+",
        ],
        include_lowest=True,
    )

    # -----------------------------------------------------
    # ANALYZE GROUPS
    # -----------------------------------------------------

    analysis = (
        results
        .groupby(
            "Confidence_Group",
            observed=False,
        )["Next_Day_Return"]
        .agg(
            [
                "count",
                "mean",
                "median",
            ]
        )
    )

    print()

    print("=" * 60)
    print("PREDICTION PROBABILITY ANALYSIS")
    print("=" * 60)

    print()

    print(analysis)

    print()

    print(
        "Higher probability groups should ideally "
        "show stronger average returns."
    )


if __name__ == "__main__":

    analyze_probabilities()