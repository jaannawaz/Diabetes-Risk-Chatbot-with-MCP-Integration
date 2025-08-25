# MCP Server

Labs and guidelines tools (mock data) exposed as HTTP endpoints to emulate MCP tools.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r mcp_server/requirements.txt
```

## Run
```bash
uvicorn mcp_server.app:app --reload --port 8002
```

## Tools
- `POST /tools/labs.getLatestHbA1c` → { value, unit, date }
- `GET /tools/guidelines.lookup?topic=...` → { summary, source, url }
- `POST /tools/translate.detectAndTranslate` (optional) → echo text and target

Security: tool allowlist, simple per-tool rate limits, no secrets in logs, caching for guidelines.
