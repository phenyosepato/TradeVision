"""
===========================================================
TradeVision AI

Random Forest Hyperparameter Tuning
===========================================================
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)


def tune_random_forest():

    X, y = prepare_dataset("GC=F")

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    parameter_grid = {

        "n_estimators": [100, 200],

        "max_depth": [4, 6, 8],

        "min_samples_leaf": [1, 2, 5],

        "max_features": ["sqrt", "log2"],

    }

    tscv = TimeSeriesSplit(n_splits=5)

    model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
    )

    search.fit(
        X_train,
        y_train,
    )

    print()

    print("=" * 60)
    print("RANDOM FOREST TUNING")
    print("=" * 60)

    print()

    print("BEST PARAMETERS")

    print(search.best_params_)

    print()

    print(f"Best CV Accuracy: {search.best_score_:.4f}")

    return search.best_estimator_


if __name__ == "__main__":

    tune_random_forest()