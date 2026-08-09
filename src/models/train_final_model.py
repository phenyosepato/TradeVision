"""
===========================================================
TradeVision AI

Final Production Model
===========================================================
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.prepare_data import (
    prepare_dataset,
)

from src.models.save_model import save_model


def train_final_model():

    X, y = prepare_dataset("GC=F")

    model = Pipeline(
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

    model.fit(
        X,
        y,
    )

    save_model(
        model,
        "tradevision_final.pkl",
    )

    print()

    print("=" * 60)
    print("FINAL MODEL")
    print("=" * 60)

    print()

    print(
        "Model: Scaled Logistic Regression"
    )

    print(
        "Training completed successfully."
    )


if __name__ == "__main__":

    train_final_model()