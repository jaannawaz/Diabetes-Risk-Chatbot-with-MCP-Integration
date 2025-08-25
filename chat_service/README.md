# Chat Service (Node/Express)

## Setup
- cd chat_service
- npm install

## Run
```bash
MODEL_URL=http://127.0.0.1:8001 MCP_URL=http://127.0.0.1:8002 npm run start
```

## Endpoint
- POST `/chat` body example:
```json
{
  "message": "hi",
  "language": "en",
  "features": {
    "gender": "Male",
    "age": 45,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 27.5,
    "HbA1c_level": 6.0,
    "blood_glucose_level": 145
  },
  "patient_id": "P001"
}
```

Returns localized text plus a structured block containing risk score, label, top factors, and any MCP facts.
