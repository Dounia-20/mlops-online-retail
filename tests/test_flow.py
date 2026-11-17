import pandas as pd
import pytest

from src.data.quality import DataQualityError
from src.orchestration import flow


def make_batch(n: int = 200, with_quantity: bool = True) -> pd.DataFrame:
    df = pd.DataFrame({
        "InvoiceNo": [str(500000 + i) for i in range(n)],
        "StockCode": ["85123A"] * n,
        "Description": pd.array(["WHITE HANGING HEART"] * n, dtype="string"),
        "Quantity": [6] * n,
        "InvoiceDate": pd.date_range("2011-01-01", periods=n, freq="h"),
        "UnitPrice": [2.55] * n,
        "CustomerID": [17850.0] * n,
        "Country": ["United Kingdom"] * n,
    })
    return df if with_quantity else df.drop(columns=["Quantity"])


@pytest.fixture
def batches(tmp_path, monkeypatch):
    cfg = flow.load_config()
    cfg["data"]["batches_dir"] = str(tmp_path)
    monkeypatch.setattr(flow, "load_config", lambda: cfg)
    return tmp_path


def test_quality_task_passes_on_valid_batches(batches):
    make_batch().to_parquet(batches / "batch_2011-01.parquet")
    make_batch().to_parquet(batches / "batch_2011-02.parquet")
    assert flow.quality_task.fn() == 2


def test_quality_task_blocks_invalid_batch(batches):
    make_batch().to_parquet(batches / "batch_2011-01.parquet")
    make_batch(with_quantity=False).to_parquet(batches / "batch_2011-03.parquet")
    with pytest.raises(DataQualityError, match="Colonnes manquantes"):
        flow.quality_task.fn()


def test_quality_task_fails_without_batches(batches):
    with pytest.raises(FileNotFoundError):
        flow.quality_task.fn()


def _stub_tasks(monkeypatch, calls, quality_ok):
    def make(name):
        def step():
            calls.append(name)
        return step

    def quality():
        calls.append("quality")
        if not quality_ok:
            raise DataQualityError("Colonnes manquantes : ['Quantity']")

    monkeypatch.setattr(flow, "ingest_task", make("ingest"))
    monkeypatch.setattr(flow, "quality_task", quality)
    monkeypatch.setattr(flow, "features_task", make("features"))
    monkeypatch.setattr(flow, "train_task", make("train"))
    monkeypatch.setattr(flow, "reload_api_task", make("reload"))


def test_pipeline_runs_all_steps_in_order(monkeypatch):
    calls = []
    _stub_tasks(monkeypatch, calls, quality_ok=True)
    flow.pipeline.fn()
    assert calls == ["ingest", "quality", "features", "train", "reload"]


def test_pipeline_stops_before_training_on_invalid_batch(monkeypatch):
    calls = []
    _stub_tasks(monkeypatch, calls, quality_ok=False)
    with pytest.raises(DataQualityError):
        flow.pipeline.fn()
    assert calls == ["ingest", "quality"]


def test_failure_hook_writes_alert(tmp_path, monkeypatch):
    cfg = flow.load_config()
    cfg["orchestration"]["alerts_path"] = str(tmp_path / "alerts.log")
    monkeypatch.setattr(flow, "load_config", lambda: cfg)
    monkeypatch.setattr(flow, "record_run", lambda *a, **k: None)

    class Run:
        name = "run-test"

    class State:
        message = "DataQualityError"

    flow.on_flow_failure(None, Run(), State())
    assert "ALERTE" in (tmp_path / "alerts.log").read_text(encoding="utf-8")
