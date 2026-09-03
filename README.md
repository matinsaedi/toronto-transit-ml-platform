# Toronto Transit ML Platform

An end-to-end machine learning service for predicting TTC bus delays.

I built this project to gain hands-on experience with the engineering involved in taking a machine learning model from experimentation to a production-style service. The main goal is not to develop a novel prediction model, but to work through the components that are commonly needed around a model in a real-world ML system, including training, inference, APIs, testing, containerization, persistence, monitoring, CI/CD, and cloud deployment.

The project uses public TTC bus delay data from the City of Toronto.

## Current Architecture

```text
Client
  |
  v
+----------------------+
|   FastAPI Container  |
|                      |
|  Input validation    |
|         |            |
|         v            |
|  XGBoost inference   |
+----------+-----------+
           |
           v
+----------------------+
| PostgreSQL Container |
|                      |
| Prediction history   |
+----------------------+

Managed with Docker Compose
```

FastAPI and PostgreSQL run in separate Docker containers and communicate over the Docker Compose network.

## Prediction Task

The model predicts the delay duration of a reported TTC bus incident in minutes.

Current model inputs:

- Day
- Line
- Incident code
- Direction (`Bound`)
- Month
- Hour

Target:

- `Min Delay`

The current model is an XGBoost regressor wrapped in a scikit-learn pipeline containing the required preprocessing steps.

The selected model achieved a validation MAE of approximately **13.55 minutes**.

## Project Flow

```text
TTC delay data
      |
      v
Data loading and preprocessing
      |
      v
Time-based train / validation / test split
      |
      v
Model training
      |
      v
Saved scikit-learn + XGBoost pipeline
      |
      v
FastAPI inference service
      |
      v
PostgreSQL prediction logging
      |
      v
Docker + Docker Compose
```

## Tech Stack

- Python 3.12
- pandas
- scikit-learn
- XGBoost
- FastAPI
- Pydantic
- PostgreSQL
- psycopg
- Docker
- Docker Compose
- pytest

## API

The service currently provides two endpoints.

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

### Predict Delay

```http
POST /predict
```

Example request:

```json
{
  "day": "Wednesday",
  "line": "102 MARKHAM ROAD",
  "code": "MFDV",
  "bound": "N",
  "month": 8,
  "hour": 17
}
```

Example response:

```json
{
  "predicted_delay_minutes": 95.1578
}
```

Each successful prediction is also stored in PostgreSQL together with the input features and request timestamp.

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/matinsaedi/toronto-transit-ml-platform.git
cd toronto-transit-ml-platform
```

### 2. Create a development environment

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -e ".[dev]"
```

### 3. Model artifact

The trained model is loaded from:

```text
artifacts/models/xgb_pipeline.joblib
```

The model artifact and raw dataset are not committed to the repository. A local model artifact is therefore required before starting the inference service.

### 4. Run with Docker Compose

```bash
docker compose up --build
```

This starts:

- the FastAPI service on port `8000`
- PostgreSQL on port `5432`

The interactive API documentation is available at:

```text
http://localhost:8000/docs
```

## Testing

API tests are implemented using `pytest` and FastAPI's `TestClient`.

Run the tests from the project root:

```bash
pytest
```

The current tests cover:

- health endpoint
- valid prediction requests
- request validation for invalid inputs

## Project Structure

```text
toronto-transit-ml-platform/
├── src/
│   └── toronto_transit_ml_platform/
│       ├── api.py
│       ├── data.py
│       ├── database.py
│       ├── features.py
│       ├── model_io.py
│       ├── predict.py
│       └── train.py
│
├── tests/
│   └── test_api.py
│
├── scripts/
├── Dockerfile
├── compose.yaml
├── pyproject.toml
└── README.md
```

## Current Status

Completed:

- Data exploration and feature selection
- Time-based dataset splitting
- Baseline model comparison
- XGBoost training pipeline
- Model serialization and loading
- FastAPI inference API
- Pydantic request validation
- API tests
- Dockerized inference service
- PostgreSQL prediction persistence
- Multi-container setup with Docker Compose

Planned:

- Basic service and model monitoring
- CI/CD with GitHub Actions
- AWS deployment
- Final documentation and architecture cleanup

## Motivation

Most of my previous machine learning work has focused on model development, deep learning, computer vision, and research. I started this project to work more directly with the engineering involved in building and deploying a complete ML service.

I wanted to go beyond training a model in a notebook and understand how the surrounding pieces fit together: organizing training and inference code, exposing predictions through an API, validating requests, testing the service, storing prediction records, containerizing the application, connecting multiple services, monitoring the system, and deploying it to the cloud.

I have intentionally kept the scope manageable so that I can implement and understand each part of the system rather than relying on a large framework to handle the full workflow.
