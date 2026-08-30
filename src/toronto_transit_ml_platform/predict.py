from pathlib import Path
import os
import pandas as pd

from toronto_transit_ml_platform.model_io import load_model

MODEL_PATH = Path(
                  os.getenv(
                            "MODEL_PATH",
                            "artifacts/models/xgb_pipeline.joblib",
                  )
)

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
