import joblib

from src.config.settings import MODEL_DIR


def load_model(filename: str):

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    return joblib.load(model_path)