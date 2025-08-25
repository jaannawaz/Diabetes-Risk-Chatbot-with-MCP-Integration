import csv
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MOCK_DIR = ROOT / "mcp_server" / "mock"
SCHEMAS_DIR = ROOT / "mcp_server" / "schemas"

ALLOWED_TOOLS = {
    "labs.getLatestHbA1c",
    "guidelines.lookup",
    "translate.detectAndTranslate",
    "genomics.lookup",
}

# simple per-tool rate limiting (requests per minute) per client id
RATE_LIMIT_PER_MINUTE = 30
_RATE_BUCKETS: Dict[str, Dict[str, float]] = {}


def _client_id(req: Request) -> str:
    return req.headers.get("X-Client-Id") or req.client.host or "global"


def _rate_limit(req: Request, tool_name: str) -> None:
    key = f"{_client_id(req)}::{tool_name}"
    now = time.time()
    bucket = _RATE_BUCKETS.get(key, {"window_start": now, "count": 0})
    # reset window after 60s
    if now - bucket["window_start"] >= 60:
        bucket = {"window_start": now, "count": 0}
    bucket["count"] += 1
    _RATE_BUCKETS[key] = bucket
    if bucket["count"] > RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded for this tool. Try again later.")


class HbA1cRequest(BaseModel):
    patient_id: str = Field(..., description="Non-PHI demo identifier")


class TranslateRequest(BaseModel):
    text: str
    target: str = Field(..., pattern="^(en|ar)$")


@lru_cache(maxsize=1)
def _load_guidelines() -> Dict[str, Dict[str, str]]:
    path = MOCK_DIR / "guidelines.json"
    with open(path, "r") as f:
        return json.load(f)


def _labs_lookup_hba1c(patient_id: str) -> Optional[Dict[str, str]]:
    path = MOCK_DIR / "labs.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("patient_id") == patient_id:
                return {
                    "value": row.get("value"),
                    "unit": row.get("unit"),
                    "date": row.get("date"),
                }
    return None


app = FastAPI(title="MCP Tools Server", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schemas")
def schemas():
    path = SCHEMAS_DIR / "tools.json"
    if not path.exists():
        raise HTTPException(status_code=500, detail="Tool schemas not found")
    with open(path) as f:
        return json.load(f)


@app.post("/tools/labs.getLatestHbA1c")
def labs_get_latest_hba1c(req: Request, payload: HbA1cRequest):
    tool_name = "labs.getLatestHbA1c"
    if tool_name not in ALLOWED_TOOLS:
        raise HTTPException(status_code=403, detail="Tool not allowed")
    _rate_limit(req, tool_name)
    rec = _labs_lookup_hba1c(payload.patient_id)
    if not rec:
        raise HTTPException(status_code=404, detail="HbA1c not found for patient")
    return rec


@app.get("/tools/guidelines.lookup")
def guidelines_lookup(req: Request, topic: str):
    tool_name = "guidelines.lookup"
    if tool_name not in ALLOWED_TOOLS:
        raise HTTPException(status_code=403, detail="Tool not allowed")
    _rate_limit(req, tool_name)
    data = _load_guidelines()
    item = data.get(topic.lower())
    if not item:
        raise HTTPException(status_code=404, detail="Guideline topic not found")
    return item


@app.post("/tools/translate.detectAndTranslate")
def translate_detect_and_translate(req: Request, payload: TranslateRequest):
    tool_name = "translate.detectAndTranslate"
    if tool_name not in ALLOWED_TOOLS:
        raise HTTPException(status_code=403, detail="Tool not allowed")
    _rate_limit(req, tool_name)
    # placeholder: return text unchanged, echo target; digits not normalized by design
    return {"text": payload.text, "target": payload.target, "note": "demo stub"}


@app.get("/tools/genomics.lookup")
def genomics_lookup(req: Request, variant_id: str):
    tool_name = "genomics.lookup"
    if tool_name not in ALLOWED_TOOLS:
        raise HTTPException(status_code=403, detail="Tool not allowed")
    _rate_limit(req, tool_name)
    path = MOCK_DIR / "genomics.json"
    with open(path, 'r') as f:
        data = json.load(f)
    item = data.get(variant_id)
    if not item:
        raise HTTPException(status_code=404, detail="Variant not found")
    return {"variant_id": variant_id, **item}


