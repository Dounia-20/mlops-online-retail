import pytest
from fastapi.testclient import TestClient

from src.api import db, main

FEATURES = ["qty_lag_1", "qty_lag_2", "qty_lag_4", "qty_roll_mean", "price_lag_1", "week_of_year", "month"]
VALID = {"qty_lag_1": 120, "qty_lag_2": 100, "qty_lag_4": 90, "qty_roll_mean": 105,
         "price_lag_1": 1.25, "week_of_year": 20, "month": 5}


class FakeModel:
    def predict(self, X):
        return [42.0]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(main, "load_champion", lambda: None)
    monkeypatch.setattr(main, "refresh_if_needed", lambda: None)
    monkeypatch.setitem(main.state, "model", None)
    monkeypatch.setitem(main.state, "version", None)
    monkeypatch.setitem(main.state, "features", None)
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def model_loaded(monkeypatch):
    monkeypatch.setitem(main.state, "model", FakeModel())
    monkeypatch.setitem(main.state, "version", "7")
    monkeypatch.setitem(main.state, "features", FEATURES)


def test_health_is_degraded_without_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"


def test_predict_without_model_returns_503_and_is_logged(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 503
    assert client.get("/stats").json()["predictions_by_status"] == {"error": 1}


def test_invalid_input_returns_422_and_is_logged(client, model_loaded):
    r = client.post("/predict", json={**VALID, "qty_lag_1": -5})
    assert r.status_code == 422
    assert client.get("/stats").json()["predictions_by_status"] == {"invalid_input": 1}


def test_predict_returns_prediction_and_model_version(client, model_loaded):
    r = client.post("/predict", json=VALID)
    body = r.json()
    assert r.status_code == 200
    assert body["prediction"] == 42.0
    assert body["model_version"] == "7"
    assert body["prediction_id"] == 1


def test_feedback_is_stored_for_a_known_prediction(client, model_loaded):
    pid = client.post("/predict", json=VALID).json()["prediction_id"]
    r = client.post("/feedback", json={"prediction_id": pid, "actual_quantity": 260})
    assert r.status_code == 200
    assert client.get("/stats").json()["feedbacks"] == 1


def test_feedback_on_unknown_prediction_returns_404(client):
    r = client.post("/feedback", json={"prediction_id": 9999, "actual_quantity": 1})
    assert r.status_code == 404
