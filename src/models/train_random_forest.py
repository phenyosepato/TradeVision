"""
===========================================================
TradeVision AI

Train Random Forest
===========================================================
"""

from sklearn.ensemble import RandomForestClassifier
from src.models.evaluate import evaluate_model
from src.models.save_model import save_model

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)


def train_random_forest():

    X, y = prepare_dataset("GC=F")

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print()

    print("=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    save_model(
        model,
        "random_forest.pkl",
    )

    return model   

if __name__ == "__main__":

    train_random_forest()