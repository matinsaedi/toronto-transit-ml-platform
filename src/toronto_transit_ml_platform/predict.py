from pathlib import Path

import pandas as pd

from toronto_transit_ml_platform.model_io import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts/models/xgb_pipeline.joblib"

model = load_model(MODEL_PATH)


def predict_delay(
    day: str,
    line: str,
    code: str,
    bound: str,
    month: int,
    hour: int,
) -> float:
    sample = pd.DataFrame(
        [
            {
                "Day": day,
                "Line": line,
                "Code": code,
                "Bound": bound,
                "month": month,
                "hour": hour,
            }
        ]
    )

    prediction = model.predict(sample)

    return float(prediction[0])
