"""
===========================================================
TradeVision AI

Model Comparison
===========================================================
"""

import pandas as pd

from src.config.settings import PROCESSED_DATA_DIR


def create_model_comparison():

    results = {
        "Model": [
            "Naive Baseline",
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Tuned Random Forest",
            "Reduced Feature Random Forest",
            "XGBoost",
        ],
        "Accuracy": [
            0.5554,
            0.5567,
            0.4647,
            0.4295,
            0.4484,
            0.4849,
            0.4421,
        ],
    }

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by="Accuracy",
        ascending=False,
    )

    print()

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print()

    print(comparison.to_string(index=False))

    print()

    best_model = comparison.iloc[0]

    print(
        f"Best Accuracy: "
        f"{best_model['Model']}"
    )

    print(
        f"Accuracy: "
        f"{best_model['Accuracy']:.4f}"
    )

    output_file = (
        PROCESSED_DATA_DIR
        / "model_comparison.csv"
    )

    comparison.to_csv(
        output_file,
        index=False,
    )

    print()

    print("Comparison saved to:")

    print(output_file)


if __name__ == "__main__":

    create_model_comparison()