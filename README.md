# Dynamic Pricing System for Online Stores

Production-structured IEEE-demo ready full-stack system using **FastAPI + ML + Q-learning**.

## Features
- `/predict` REST endpoint with preprocessing, feature engineering, and dynamic price optimization.
- Model training/comparison across **Random Forest**, **Gradient Boosting**, and **XGBoost** with cross-validation and hyperparameter tuning.
- Business constraints: min/max thresholds and anti-volatility cap.
- Live dashboard with KPI monitoring and Chart.js visualization.
- Swagger docs at `/docs`.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Docker
```bash
docker compose up --build
```

Open: `http://localhost:8000`
