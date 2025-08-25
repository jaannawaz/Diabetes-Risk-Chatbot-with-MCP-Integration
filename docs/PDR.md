# Project Definition Report (PDR)

## Title
Bilingual Diabetes Risk Chatbot with MCP Integration

## Purpose
Build a diabetes risk assistant that predicts risk from structured inputs, explains results clearly, and can fetch labs/guidelines via an MCP server. Output to users in English and Arabic.

## Scope and Constraints
- Language handling: assume numeric inputs are standard digits (no Arabic digit normalization)
- Privacy: no PHI persistence, secrets server-side only, no secrets in source control/logs
- Scope: one local CSV dataset, no EHR integration, demo-quality services

## Dataset
- Path: `./diabetes_prediction_dataset.csv`
- Target column: `Outcome`
- Split: train 0.70, val 0.15, test 0.15 (stratified, seed=42)
- Preprocessing: median imputation, scaling for SVM/MLP, categorical encoding if present

## Feature Schema (initial)
```
{
  "glucose_mg_dl": number,
  "bmi": number,
  "age_years": number,
  "sex": "male"|"female",
  "pregnancies": number,
  "blood_pressure": number,
  "skin_thickness": number,
  "insulin": number,
  "diabetes_pedigree": number
}
```

## Objectives and Deliverables
1. Data Collection & Preparation
   - Deliverables: data card (Markdown), feature schema (JSON/YAML), EDA summary
2. ML Training & Model Selection
   - Deliverables: metrics.csv, roc.png, confusion_matrix.png, shap_summary.png, local_explanations.png, pdp_glucose.png, pdp_bmi.png, model_selection_notes.md
3. Model Service (API Layer)
   - Deliverables: FastAPI app with /predict, /explain endpoints, OpenAPI docs, unit tests
4. MCP Server Integration
   - Deliverables: MCP server with labs/guidelines (optional translate), tool schemas/README, mock datasets
5. Conversational Layer (Chat Backend)
   - Deliverables: Chat API with POST /chat, EN/AR templates, safety copy
6. Web Interface & Validation
   - Deliverables: React app, screenshots for slides, clinician report mock

## Architecture Overview
- ML pipeline trains Logistic Regression, Random Forest, XGBoost, SVM (RBF), and MLP; select best by validation metrics with target recall ≥ 0.90 and specificity ≈ ≥ 0.80. Prefer XGBoost for serving; optional stacking ensemble documented.
- Model Service (FastAPI) exposes:
  - `POST /predict` → { risk_score, risk_label, top_factors[], echo_features }
  - `GET /explain/global` → paths to SHAP/importance assets
  - `POST /explain/local` → top-k contributions for one case
  - `GET /health`
- MCP Server tools:
  - `labs.getLatestHbA1c(patient_id)` → { value, unit, date }
  - `guidelines.lookup(topic)` → { summary, source, url } with caching
  - Optional: `translate.detectAndTranslate(text, target)`
- Chat Service (Node/Express): orchestrates Model and MCP, normalizes inputs to feature schema, localizes output (EN/AR), includes safety disclaimer and escalation triggers.
- Web UI (React): consent modal, language toggle (Auto/EN/AR) with RTL, chat + results panel.

## Security & Privacy
- No PHI persistence; mock data only for labs; redact logs
- Secrets stored server-side via environment variables; never committed
- Tool allowlist and per-tool rate limits for MCP server

## Testing & Validation
- Unit tests: Model Service payloads/schemas/threshold edges; MCP tools happy/not-found/cache; Chat safety/language/error surfaces
- Integration tests: Chat→Model, Chat→MCP
- E2E: high-risk EN flow, low-risk EN/AR flow, labs query, safety escalation
- Non-functional: local round-trip < 2s; configs documented

## Artifacts
- `artifacts/metrics.csv`, `roc.png`, `confusion_matrix.png`, `shap_summary.png`, `local_explanations.png`, `pdp_glucose.png`, `pdp_bmi.png`

## Risks & Mitigations
- Class imbalance → stratified split, threshold tuning, AUC/F1 monitoring
- Overfitting → validation split, cross-validation as needed, SHAP sanity checks
- Explainability drift → lock features, document preprocessing, version artifacts

## Timeline & Run Order
1. Create repo layout and READMEs
2. Implement ML pipeline, train models, choose best model and threshold, export artifacts
3. Implement Model Service and tests
4. Implement MCP Server and tests
5. Implement Chat Service with language handling and safety; integrate Model/MCP
6. Build Web UI; validate EN/AR behavior; export slide assets

## Project Extension: Enhanced Diabetes Assistant (Objectives 7–11)
Title: Enhanced Diabetes Assistant with Multi-Modal Inputs and Clinical Integration
Purpose: Upgrade the chatbot into an MCP-powered clinical assistant supporting multi-modal data, precision medicine hooks, patient coaching, and clinician reporting.

### Objective 7: Multi-Modal Data Input
- Upload and parse labs (CSV/PDF) for HbA1c, fasting glucose (mock parser)
- Optional wearable/IoT data connector (steps, HR, activity) as risk modifiers (stub)
- Voice input (EN/AR) with Whisper/Azure Speech to capture symptoms → structured schema

### Objective 8: Explainability for Patients and Clinicians
- Dual SHAP modes: simple language (patient) vs detailed charts (clinician)
- Progress tracking of risk over time; trend visualization
- Layman-friendly explanations: e.g., BMI and glucose impact

### Objective 9: Precision Medicine Hooks
- Placeholder for genomic variants (e.g., TCF7L2; population-specific notes)
- Mock MCP tool: genomics.lookup(variant_id)
- Document future pipeline: VCF → annotation → risk adjustment

### Objective 10: Behavioral Coaching Layer
- Tailored lifestyle recommendations (EN/AR)
- Sentiment detection and tone adjustment
- Follow-up reminder mock (non-persistent)

### Objective 11: Clinician-Integrated Reporting
- One-page PDF/HTML with inputs, risk, top factors, guidelines
- Mock FHIR/HL7 export adapter (simulated)
- Dual outputs: JSON + PDF (EN/AR)

Priority for demo: 7 (multi-modal), 8 (explainability), 11 (reporting), with 10 and 9 as stretch.
