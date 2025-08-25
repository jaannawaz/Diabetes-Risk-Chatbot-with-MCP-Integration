# Bilingual Diabetes Risk Chatbot with MCP Integration

## Overview
A demo-quality, privacy-conscious assistant that:
- Predicts diabetes risk from structured inputs
- Explains results with global and local interpretability
- Fetches labs/guidelines via an MCP server
- Responds in English and Arabic

## Constraints
- No PHI persistence; keep secrets server-side only
- One local CSV dataset; no EHR integration
- Numeric inputs arrive as standard digits

## Dataset
- Path: `./diabetes_prediction_dataset.csv`
- Target column: `Outcome`
- Split: train 70%, val 15%, test 15% (stratified, seed=42)

## Services
- Model Service (FastAPI): `/predict`, `/explain/global`, `/explain/local`, `/health`
- MCP Server: `labs.getLatestHbA1c`, `guidelines.lookup`, optional `translate`
- Chat Service (Node/Express): `/chat` with language hint `auto|en|ar`
- Web UI (React): EN/AR with RTL support

## Extension (Objectives 7–11)
Title: Enhanced Diabetes Assistant with Multi-Modal Inputs and Clinical Integration
- Multi-Modal Input: Upload labs (CSV/PDF mock), optional wearable data, EN/AR voice input → structured schema
- Explainability: Dual SHAP modes (patient-friendly vs clinician charts), risk history trends
- Precision Hooks: Mock `genomics.lookup(variant_id)` via MCP; document future VCF→annotation pipeline
- Coaching: Tailored EN/AR recommendations, sentiment-aware tone, reminder mocks
- Clinician Reporting: One-page PDF/HTML + JSON; mock FHIR export adapter

Priority for demo: Multi-modal input, Explainability, Clinician reporting; coaching and genomics as stretch.

## Run Order
1. Implement ML pipeline, train models, export artifacts
2. Implement Model Service and tests
3. Implement MCP Server and tests
4. Implement Chat Service and integrate
5. Build Web UI and validate E2E

## Local Development
- Create an isolated environment per service (e.g., conda, venv, or nvm)
- Do not commit secrets; use `.env` (untracked)

### ML
- Python 3.11 recommended
- Outputs: `artifacts/metrics.csv`, `artifacts/*.png`, and serialized model

### Model Service
- FastAPI + Uvicorn
- Loads best model and threshold from `artifacts/`

### MCP Server
- Exposes labs/guidelines tools over MCP protocol (mock data, cached)

### Chat Service
- Node/Express orchestrating Model and MCP; server-side OpenAI key

### Web
- React app consuming Chat API; language toggle and RTL for AR

## Licensing & Credits
- Dataset license TBD; include in `docs/data_card.md`

## Repository Structure
```
ml/
model_service/
mcp_server/
chat_service/
web/
artifacts/
slides/
docs/
```
