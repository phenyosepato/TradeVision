"""
===========================================================
TradeVision AI

Feature Selection Experiment
===========================================================
"""

from sklearn.ensemble import RandomForestClassifier

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)

from src.models.evaluate import evaluate_model


SELECTED_FEATURES = [
    "Log_Return",
    "Volatility",
    "Daily_Return",
    "Volume",
    "ATR",
    "MACD",
    "SMA_200",
    "MACD_Signal",
    "RSI",
    "SMA_50",
]


def run_feature_selection():

    X, y = prepare_dataset("GC=F")

    X = X[SELECTED_FEATURES]

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print()

    print("=" * 60)
    print("REDUCED FEATURE RANDOM FOREST")
    print("=" * 60)

    print()

    print("Features used:")

    for feature in SELECTED_FEATURES:
        print(f"- {feature}")

    evaluate_model(
        model,
        X_test,
        y_test,
    )


if __name__ == "__main__":

    run_feature_selection()