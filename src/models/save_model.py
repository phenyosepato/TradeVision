"""
===========================================================
TradeVision AI

Save Trained Models
===========================================================
"""

import joblib

from src.config.settings import MODEL_DIR


def save_model(
    model,
    filename,
):
    """
    Save a trained model.
    """

    path = MODEL_DIR / filename

    joblib.dump(
        model,
        path,
    )

    print()

    print(f"Model saved to:\n{path}")