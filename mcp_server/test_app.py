from fastapi.testclient import TestClient
from mcp_server.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_guidelines_lookup():
    r = client.get("/tools/guidelines.lookup", params={"topic": "hba1c"})
    assert r.status_code == 200
    data = r.json()
    assert "summary" in data


def test_labs_get_latest_hba1c():
    r = client.post("/tools/labs.getLatestHbA1c", json={"patient_id": "P001"}, headers={"X-Client-Id": "demo"})
    assert r.status_code == 200
    data = r.json()
    assert data["unit"] == "%"


