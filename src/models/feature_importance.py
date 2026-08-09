"""
===========================================================
TradeVision AI

Random Forest Feature Importance
===========================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)


def analyze_feature_importance():

    X, y = prepare_dataset("GC=F")

    X_train, X_test, y_train, y_test = split_dataset(X, y)

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

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
    )

    importance = importance.sort_values(
        ascending=False
    )

    print()

    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    print()

    print(importance)

    plt.figure(figsize=(10, 8))

    importance.sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.xlabel(
        "Importance"
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    analyze_feature_importance()