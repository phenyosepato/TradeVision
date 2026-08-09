"""
===========================================================
TradeVision AI

Permutation Feature Importance
===========================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)


def analyze_permutation_importance():

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

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importance = pd.Series(
        result.importances_mean,
        index=X.columns,
    )

    importance = importance.sort_values(
        ascending=False
    )

    print()

    print("=" * 60)
    print("PERMUTATION FEATURE IMPORTANCE")
    print("=" * 60)

    print()

    print(importance)

    importance.to_csv(
    "reports/permutation_importance.csv",
    header=["importance"],
    )

    print()
    print("Permutation importance saved to:")
    print("reports/permutation_importance.csv")

    plt.figure(figsize=(10, 8))

    importance.sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Permutation Feature Importance"
    )

    plt.xlabel(
        "Mean Importance"
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    analyze_permutation_importance()