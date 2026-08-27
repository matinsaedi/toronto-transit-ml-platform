from pathlib import Path

import joblib


def save_model(model, path):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, path)


def load_model(path):
    path = Path(path)

    return joblib.load(path)
