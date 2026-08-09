"""
===========================================================
TradeVision AI

Decision Tree Hyperparameter Tuning
===========================================================
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)


def tune_decision_tree():

    X, y = prepare_dataset("GC=F")

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    parameter_grid = {

        "max_depth": [3, 5, 7, 10],

        "min_samples_split": [2, 5, 10],

        "min_samples_leaf": [1, 2, 5],

    }

    tscv = TimeSeriesSplit(n_splits=5)
    model = DecisionTreeClassifier(
        random_state=42,
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
    print("BEST PARAMETERS")
    print("=" * 60)

    print(search.best_params_)

    print()

    print(f"Best CV Accuracy: {search.best_score_:.4f}")

    return search.best_estimator_


if __name__ == "__main__":

    tune_decision_tree()