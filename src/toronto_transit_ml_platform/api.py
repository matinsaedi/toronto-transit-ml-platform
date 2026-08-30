from fastapi import FastAPI
from pydantic import BaseModel, Field
from toronto_transit_ml_platform.predict import predict_delay

class PredictionRequest(BaseModel):
    day: str
    line: str
    code: str
    bound: str
    month: int = Field(ge=1, le=12)
    hour: int = Field(ge=0, le=23)

app = FastAPI()

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
    
    return {"predicted_delay_minutes": prediction}
