"""
===========================================================
TradeVision AI

Tune Logistic Regression
===========================================================
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.prepare_data import prepare_dataset


def tune_logistic():

    X, y = prepare_dataset("GC=F")

    pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    parameter_grid = {
        "logistic__C": [
            0.001,
            0.01,
            0.1,
            1,
            10,
            100,
        ]
    }

    grid_search = GridSearchCV(
        pipeline,
        parameter_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search.fit(X, y)

    print()

    print("=" * 60)
    print("LOGISTIC REGRESSION TUNING")
    print("=" * 60)

    print()

    print("BEST PARAMETERS")

    print(
        grid_search.best_params_
    )

    print()

    print(
        f"Best CV Accuracy: "
        f"{grid_search.best_score_:.4f}"
    )


if __name__ == "__main__":

    tune_logistic()