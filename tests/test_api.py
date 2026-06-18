"""Test suite for FastAPI endpoints of Conveyor Inspection."""

import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["camera_ready"] is True
    assert data["rejection_actuator_ready"] is True


def test_api_inspect_synthetic_puncture(client):
    payload = {"defect_type": "puncture", "seed": 42, "unit_id": "UNIT_991"}
    res = client.post("/inspect/synthetic", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["pass_status"] == "REJECT"
    assert data["rejection_actuator_triggered"] is True
    assert data["primary_defect_type"] == "Puncture / Leak"
    assert len(data["detected_defects"]) >= 1


def test_api_inspect_synthetic_normal(client):
    payload = {"defect_type": "normal", "seed": 42, "unit_id": "UNIT_992"}
    res = client.post("/inspect/synthetic", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["pass_status"] == "PASS"
    assert data["rejection_actuator_triggered"] is False


def test_api_conveyor_stream_batch(client):
    payload = {"batch_size": 20, "defect_probability": 0.25}
    res = client.post("/conveyor/stream-batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_units_inspected"] == 20
    assert data["units_passed"] + data["units_rejected"] == 20
    assert data["simulated_ppm"] > 100.0
