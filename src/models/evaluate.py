"""
===========================================================
TradeVision AI

Model Evaluation
===========================================================
"""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate a trained model.
    """

    predictions = model.predict(X_test)

    print()

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")

    print("\nConfusion Matrix\n")

    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report\n")

    print(classification_report(y_test, predictions))