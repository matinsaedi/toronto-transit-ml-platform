# Toronto Transit ML Platform

An end-to-end machine learning service for predicting TTC bus delays.

I built this project to gain hands-on experience with the engineering involved in taking a machine learning model from experimentation to a production-style service. The goal is not to develop a novel prediction model, but to work through the surrounding components commonly needed in a real-world ML system: training, inference, APIs, testing, containerization, persistence, monitoring, CI/CD, model artifact storage, and cloud deployment.

The project uses public TTC bus delay data from the City of Toronto.

## Architecture

```text
                         GitHub
                            |
                            | push to main
                            v
                    +----------------+
                    | GitHub Actions |
                    |                |
                    | Tests          |
                    | Docker build   |
                    | CD deployment  |
                    +--------+-------+
                             |
                         AWS OIDC
                             |
                             v
                      AWS Systems Manager
                             |
                             v
+----------------------------------------------------------+
|                         AWS EC2                          |
|                                                          |
|   +----------------------+                               |
|   |   FastAPI Container  |                               |
|   |                      |                               |
|   |  Input validation    |                               |
|   |         |            |                               |
|   |         v            |                               |
|   |  XGBoost inference   |                               |
|   +----------+-----------+                               |
|              |                                           |
|              v                                           |
|   +----------------------+      +----------------------+  |
|   | PostgreSQL Container |      | Prometheus Container |  |
|   |                      |      |                      |  |
|   | Prediction history   |      | API metrics          |  |
|   +----------------------+      +----------------------+  |
|                                                          |
|              Docker Compose                              |
+----------------------------------------------------------+
                   ^
                   |
                   | model artifact
                   |
              Amazon S3
```

The production service runs on an AWS EC2 instance using Docker Compose.

The trained model is stored separately in a private Amazon S3 bucket rather than committed to Git. The EC2 instance accesses the model using an IAM role.

PostgreSQL stores prediction records, while Prometheus collects service metrics exposed by the FastAPI application.

Production services are not directly exposed to the public internet. The API and Prometheus are bound to the EC2 localhost interface and can be accessed securely through an SSH tunnel.

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

## ML Pipeline

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
Model training and evaluation
      |
      v
Serialized scikit-learn + XGBoost pipeline
      |
      v
Amazon S3 model storage
      |
      v
FastAPI inference service
      |
      +-----------> PostgreSQL prediction logging
      |
      +-----------> Prometheus metrics
```

## Tech Stack

### Machine Learning

- Python 3.12
- pandas
- scikit-learn
- XGBoost
- joblib

### API and Persistence

- FastAPI
- Pydantic
- PostgreSQL
- psycopg

### Infrastructure and Monitoring

- Docker
- Docker Compose
- Prometheus
- AWS EC2
- Amazon S3
- AWS IAM
- AWS Systems Manager

### Testing and CI/CD

- pytest
- GitHub Actions
- GitHub OIDC authentication with AWS

## API

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

### Metrics

```http
GET /metrics
```

The service exposes Prometheus-compatible metrics including:

- successful prediction requests
- prediction latency
- HTTP error responses

Prometheus periodically scrapes this endpoint and stores the collected metrics.

## Running Locally

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

The inference service expects the trained model at:

```text
artifacts/models/xgb_pipeline.joblib
```

Model artifacts and raw datasets are intentionally excluded from Git.

A local copy of the trained model is therefore required before running real-model inference locally.

### 4. Run with Docker Compose

```bash
docker compose up --build
```

This starts the local development stack:

- FastAPI
- PostgreSQL
- Prometheus

The FastAPI documentation is available at:

```text
http://localhost:8000/docs
```

Prometheus is available at:

```text
http://localhost:9090
```

## Production Deployment

The production stack runs on an Ubuntu EC2 instance using:

```text
compose.prod.yaml
```

Production configuration differs from local development in several ways:

- PostgreSQL is not published on a host port.
- FastAPI is bound to `127.0.0.1:8000`.
- Prometheus is bound to `127.0.0.1:9090`.
- PostgreSQL credentials are supplied through a private `.env.prod` file.
- The trained model is mounted into the API container from the EC2 host.
- PostgreSQL and Prometheus use persistent Docker volumes.

The `.env.prod` file and model artifacts are excluded from Git.

The model artifact is stored in a private S3 bucket and can be downloaded by the EC2 instance using its IAM role without storing AWS access keys on the server.

## CI/CD

GitHub Actions provides both continuous integration and continuous deployment.

### Continuous Integration

For pushes and pull requests targeting `main`, the CI workflow:

1. Checks out the repository.
2. Installs Python and project dependencies.
3. Runs the lightweight API test suite.
4. Builds the Docker image.

The CI test suite excludes tests that require the trained model artifact because that artifact is intentionally not stored in Git.

### Continuous Deployment

For successful pushes to `main`, the deployment job runs after CI completes.

```text
Push to main
      |
      v
API tests
      |
      v
Docker build
      |
      v
AWS authentication through OIDC
      |
      v
AWS Systems Manager
      |
      v
EC2 deployment
      |
      v
Health check
```

GitHub Actions authenticates to AWS using OpenID Connect (OIDC), avoiding long-lived AWS access keys in GitHub.

AWS Systems Manager sends the deployment command to EC2. The server then:

1. Pulls the latest commit from `main`.
2. Rebuilds and restarts the Docker Compose stack.
3. Checks the `/health` endpoint.
4. Reports deployment success or failure back to GitHub Actions.

## Testing

Tests are implemented with `pytest` and FastAPI's `TestClient`.

Run the complete local test suite with:

```bash
pytest
```

Run only tests that do not require the trained model artifact with:

```bash
pytest -m "not requires_model"
```

The current tests cover:

- health endpoint
- valid prediction requests
- request validation
- metrics endpoint
- real-model inference

API tests mock model inference where appropriate so they can run independently from the trained artifact.

## Project Structure

```text
toronto-transit-ml-platform/
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── src/
│   └── toronto_transit_ml_platform/
│       ├── api.py
│       ├── data.py
│       ├── database.py
│       ├── features.py
│       ├── model_io.py
│       ├── monitoring.py
│       ├── predict.py
│       └── train.py
│
├── tests/
│   ├── test_api.py
│   └── test_model.py
│
├── prometheus/
│   └── prometheus.yml
│
├── scripts/
├── Dockerfile
├── compose.yaml
├── compose.prod.yaml
├── pyproject.toml
└── README.md
```

## Current Status

Completed:

- Data exploration and feature selection
- Time-based train / validation / test splitting
- Baseline model comparison
- XGBoost training pipeline
- Model serialization and loading
- FastAPI inference API
- Pydantic request validation
- PostgreSQL prediction persistence
- Dockerized multi-container application
- Prometheus service monitoring
- Automated API testing
- Docker build validation in CI
- Private model artifact storage in Amazon S3
- AWS EC2 deployment
- IAM-based access to AWS resources
- Secure EC2 management through AWS Systems Manager
- GitHub Actions CI/CD
- GitHub-to-AWS authentication using OIDC
- Automatic deployment to EC2 after successful CI
- Post-deployment health checking

Remaining work is primarily documentation, cleanup, and small reliability improvements.

## Motivation

Most of my previous machine learning work has focused on model development, deep learning, computer vision, and research. I started this project to gain experience with the engineering required to turn a trained model into a complete deployed service.

The project gave me hands-on experience with separating training and inference code, designing an API, validating requests, writing automated tests, persisting predictions, containerizing multiple services, collecting operational metrics, managing model artifacts outside Git, deploying to AWS, and building an automated CI/CD workflow.

I intentionally kept the infrastructure manageable so that I could implement and understand each component rather than relying on a large platform to handle the entire workflow.
