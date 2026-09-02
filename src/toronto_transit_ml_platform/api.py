from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from toronto_transit_ml_platform.predict import predict_delay
from toronto_transit_ml_platform.database import create_predictions_table, save_prediction


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

    return {"predicted_delay_minutes": prediction}
