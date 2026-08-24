from fastapi.testclient import TestClient

from app.main import app
from app.ml.model_manager import set_model_loaded

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}
    assert "X-Request-ID" in response.headers


def test_ready_endpoint_not_loaded():
    set_model_loaded(False)
    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["model_loaded"] is False


def test_ready_endpoint_loaded():
    set_model_loaded(True)
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
