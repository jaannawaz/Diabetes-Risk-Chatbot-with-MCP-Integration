# Model Service (FastAPI)

Serves predictions and explanations from the trained diabetes risk model.

## Setup
1. Ensure artifacts exist (run training): `python ml/train.py` from repo root.
2. Create a virtualenv and install deps:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r model_service/requirements.txt`

## Run
```bash
uvicorn model_service.app:app --reload --port 8001
```

## Endpoints
- `GET /health`
- `POST /predict`
- `GET /explain/global`
- `POST /explain/local`

Input schema matches dataset features: `gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level`.
