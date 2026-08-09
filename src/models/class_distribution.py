"""
===========================================================
TradeVision AI

Target Class Distribution
===========================================================
"""

from src.models.prepare_data import prepare_dataset


def analyze_class_distribution():

    X, y = prepare_dataset("GC=F")

    distribution = y.value_counts()
    percentages = y.value_counts(normalize=True) * 100

    print()

    print("=" * 60)
    print("TARGET CLASS DISTRIBUTION")
    print("=" * 60)

    print()

    for class_value in distribution.index:

        print(
            f"Class {class_value}: "
            f"{distribution[class_value]} samples "
            f"({percentages[class_value]:.2f}%)"
        )


if __name__ == "__main__":

    analyze_class_distribution()