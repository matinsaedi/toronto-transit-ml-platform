from fastapi import FastAPI, Response
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from toronto_transit_ml_platform.predict import predict_delay
from toronto_transit_ml_platform.database import create_predictions_table, save_prediction
from toronto_transit_ml_platform.monitoring import prediction_requests

class PredictionRequest(BaseModel):
    day: str
    line: str
    code: str
    bound: str
    month: int = Field(ge=1, le=12)
    hour: int = Field(ge=0, le=23)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_predictions_table()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = predict_delay(
	request.day,
	request.line,
	request.code,
	request.bound,
	request.month,
	request.hour,
	)

    save_prediction(
        request.day,
        request.line,
        request.code,
        request.bound,
        request.month,
        request.hour,
        predicted_delay_minutes=prediction
        )

    prediction_requests.inc()

    return {"predicted_delay_minutes": prediction}

@app.get("/metrics")
def metrics():
    return Response(
	content=generate_latest(),
	media_type=CONTENT_TYPE_LATEST,
    )
