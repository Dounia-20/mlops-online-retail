import json
import time

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.api import db
from src.data.quality import load_config

cfg = load_config()
tc = cfg["training"]
app = FastAPI(title="Online Retail - prévision des quantités vendues")
state = {"model": None, "version": None, "features": None}


class PredictionInput(BaseModel):
    qty_lag_1: float = Field(ge=0, description="Quantité vendue la semaine -1")
    qty_lag_2: float = Field(ge=0, description="Quantité vendue la semaine -2")
    qty_lag_4: float = Field(ge=0, description="Quantité vendue la semaine -4")
    qty_roll_mean: float = Field(ge=0, description="Moyenne glissante des 4 dernières semaines")
    price_lag_1: float = Field(gt=0, description="Prix moyen la semaine -1")
    week_of_year: int = Field(ge=1, le=53)
    month: int = Field(ge=1, le=12)


class FeedbackInput(BaseModel):
    prediction_id: int
    actual_quantity: float = Field(ge=0, description="Quantité réellement vendue")
    comment: str | None = None


def load_champion() -> None:
    mlflow.set_tracking_uri(tc["tracking_uri"])
    client = MlflowClient()
    mv = client.get_model_version_by_alias(tc["model_name"], "champion")
    path = client.download_artifacts(mv.run_id, "features.json")
    with open(path, encoding="utf-8") as f:
        state["features"] = json.load(f)["features"]
    state["model"] = mlflow.sklearn.load_model(f"models:/{tc['model_name']}@champion")
    state["version"] = mv.version


@app.on_event("startup")
def startup() -> None:
    db.init_db()
    try:
        load_champion()
    except Exception as e:
        print(f"Modèle non chargé : {e}")


@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    if request.url.path == "/predict":
        db.log_prediction(state["version"], None, None, None, "invalid_input",
                          error=str(exc.errors())[:500])
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


@app.get("/health")
def health():
    ok = state["model"] is not None
    return {"status": "ok" if ok else "degraded", "model_version": state["version"]}


@app.post("/reload")
def reload_model():
    try:
        load_champion()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Rechargement impossible : {e}")
    return {"model_version": state["version"]}


@app.post("/predict")
def predict(item: PredictionInput):
    refresh_if_needed()
    if state["model"] is None:
        db.log_prediction(None, item.model_dump(), None, None, "error", error="Aucun modèle disponible")
        raise HTTPException(status_code=503, detail="Aucun modèle disponible")
    t0 = time.perf_counter()
    try:
        row = pd.DataFrame([item.model_dump()])[state["features"]]
        pred = round(float(state["model"].predict(row)[0]), 2)
    except Exception as e:
        db.log_prediction(state["version"], item.model_dump(), None, None, "error", error=str(e)[:500])
        raise HTTPException(status_code=500, detail="Erreur pendant la prédiction")
    duration = round((time.perf_counter() - t0) * 1000, 2)
    pid = db.log_prediction(state["version"], item.model_dump(), pred, duration, "ok")
    return {"prediction_id": pid, "prediction": pred,
            "model_version": state["version"], "duration_ms": duration}


@app.post("/feedback")
def feedback(fb: FeedbackInput):
    fid = db.add_feedback(fb.prediction_id, fb.actual_quantity, fb.comment)
    if fid is None:
        raise HTTPException(status_code=404, detail="prediction_id inconnu")
    return {"feedback_id": fid}


@app.get("/stats")
def stats():
    return db.stats()


from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

REQUESTS = Counter("api_requests_total", "Requêtes HTTP", ["endpoint", "status"])
LATENCY = Histogram("api_request_duration_seconds", "Durée des requêtes", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
PREDICTIONS = Counter("api_predictions_total", "Prédictions réussies", ["model_version"])
MODEL_VERSION = Gauge("api_model_version", "Version du modèle champion chargée")
MODEL_VERSION.set_function(lambda: float(state["version"] or 0))
KNOWN = {"/predict", "/feedback", "/health", "/reload", "/stats"}


@app.middleware("http")
async def metrics_middleware(request, call_next):
    path = request.url.path if request.url.path in KNOWN else "other"
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        REQUESTS.labels(path, str(status)).inc()
        LATENCY.labels(path).observe(time.perf_counter() - t0)
        if path == "/predict" and status == 200:
            PREDICTIONS.labels(str(state["version"])).inc()


app.mount("/metrics", make_asgi_app())


from prometheus_client import REGISTRY
from prometheus_client.core import GaugeMetricFamily


class PipelineCollector:

    def collect(self):
        try:
            s = db.pipeline_stats()
        except Exception:
            return
        runs = GaugeMetricFamily("pipeline_runs", "Executions du pipeline par statut", labels=["status"])
        for status, n in s["by_status"].items():
            runs.add_metric([status], n)
        yield runs
        last = s["last"]
        if last:
            yield GaugeMetricFamily("pipeline_last_run_success", "1 si la derniere execution a reussi",
                                    value=1.0 if last["status"] == "success" else 0.0)
            yield GaugeMetricFamily("pipeline_last_run_timestamp_seconds", "Date de la derniere execution",
                                    value=float(last["ts"]))
            if last["duration_s"] is not None:
                yield GaugeMetricFamily("pipeline_last_run_duration_seconds", "Duree de la derniere execution",
                                        value=float(last["duration_s"]))


REGISTRY.register(PipelineCollector())


import threading

_refresh_lock = threading.Lock()
REFRESH_EVERY_S = 5


def refresh_if_needed() -> None:
    if time.time() - state.get("checked_at", 0.0) < REFRESH_EVERY_S:
        return
    with _refresh_lock:
        if time.time() - state.get("checked_at", 0.0) < REFRESH_EVERY_S:
            return
        state["checked_at"] = time.time()
        try:
            mlflow.set_tracking_uri(tc["tracking_uri"])
            mv = MlflowClient().get_model_version_by_alias(tc["model_name"], "champion")
            if str(mv.version) != str(state["version"]):
                load_champion()
        except Exception:
            pass

