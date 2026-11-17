import sys

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from src.training import rollback
from src.training.train import should_promote

TOLERANCE = 0.01


def test_first_model_is_promoted_without_champion():
    assert should_promote(520.8, None, TOLERANCE) is True


def test_equivalent_version_within_tolerance_is_promoted():
    assert should_promote(524.7, 520.8, TOLERANCE) is True


def test_degraded_version_is_rejected():
    assert should_promote(552.5, 520.8, TOLERANCE) is False


def test_zero_tolerance_rejects_slightly_worse_version():
    assert should_promote(524.7, 520.8, 0.0) is False


def test_model_not_beating_baseline_is_rejected_even_without_champion():
    assert should_promote(560.0, None, TOLERANCE, baseline_mae=548.2) is False


def test_model_equal_to_baseline_is_rejected():
    assert should_promote(548.2, None, TOLERANCE, baseline_mae=548.2) is False


def test_model_beating_baseline_follows_champion_rule():
    assert should_promote(520.8, None, TOLERANCE, baseline_mae=548.2) is True
    assert should_promote(520.8, 500.0, TOLERANCE, baseline_mae=548.2) is False


@pytest.fixture
def registry(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    name = "test_model"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    client.create_registered_model(name)
    for _ in range(2):
        client.create_model_version(name, source=str(tmp_path))
    client.set_registered_model_alias(name, "champion", "2")
    client.set_model_version_tag(name, "1", "status", "archived")
    client.set_model_version_tag(name, "2", "status", "champion")
    monkeypatch.setattr(rollback, "load_config",
                        lambda: {"training": {"model_name": name, "tracking_uri": uri}})
    return client, name


def test_rollback_restores_previous_version(registry, monkeypatch):
    client, name = registry
    monkeypatch.setattr(sys, "argv", ["rollback", "1"])
    rollback.main()
    assert int(client.get_model_version_by_alias(name, "champion").version) == 1
    assert client.get_model_version(name, "1").tags["status"] == "champion"
    assert client.get_model_version(name, "2").tags["status"] == "archived"


def test_rollback_to_unknown_version_keeps_champion(registry, monkeypatch):
    client, name = registry
    monkeypatch.setattr(sys, "argv", ["rollback", "99"])
    with pytest.raises(Exception):
        rollback.main()
    assert client.get_model_version(name, "2").tags["status"] == "champion"
    assert int(client.get_model_version_by_alias(name, "champion").version) == 2


def test_rollback_to_rejected_version_is_refused(registry, monkeypatch):
    client, name = registry
    client.set_model_version_tag(name, "1", "status", "rejected")
    monkeypatch.setattr(sys, "argv", ["rollback", "1"])
    with pytest.raises(SystemExit):
        rollback.main()
    assert int(client.get_model_version_by_alias(name, "champion").version) == 2
    assert client.get_model_version(name, "2").tags["status"] == "champion"
