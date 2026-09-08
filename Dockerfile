FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir .

ENV MODEL_PATH=/app/artifacts/models/xgb_pipeline.joblib

EXPOSE 8000

CMD ["uvicorn", "toronto_transit_ml_platform.api:app", "--host", "0.0.0.0", "--port", "8000"]
