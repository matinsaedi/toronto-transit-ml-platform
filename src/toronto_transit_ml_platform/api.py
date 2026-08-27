from fastapi import FastAPI
from pydantic import BaseModel
from toronto_transit_ml_platform.predict import predict_delay

class PredictionRequest(BaseModel):
    day: str
    line: str
    code: str
    bound: str
    month: int
    hour: int

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
