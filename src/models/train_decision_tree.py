"""
===========================================================
TradeVision AI

Train Decision Tree
===========================================================
"""

from sklearn.tree import DecisionTreeClassifier
from src.models.evaluate import evaluate_model
from src.models.save_model import save_model

from src.models.prepare_data import (
    prepare_dataset,
    split_dataset,
)


def train_decision_tree():

    X, y = prepare_dataset("GC=F")

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=5,
    )

    model.fit(X_train, y_train)

    print()

    print("=" * 60)
    print("DECISION TREE")
    print("=" * 60)

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    save_model(
        model,
        "decision_tree.pkl",
    )

    return model


if __name__ == "__main__":

    train_decision_tree()