"""
===========================================================
TradeVision AI

Evaluate Tuned Random Forest
===========================================================
"""

from sklearn.ensemble import RandomForestClassifier

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)

from src.models.evaluate import evaluate_model


def evaluate_tuned_random_forest():

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

    print()

    print("=" * 60)
    print("TUNED RANDOM FOREST")
    print("=" * 60)

    evaluate_model(
        model,
        X_test,
        y_test,
    )


if __name__ == "__main__":

    evaluate_tuned_random_forest()