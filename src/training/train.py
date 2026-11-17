import hashlib
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.quality import load_config

TARGET = "quantity"
NON_FEATURES = ["StockCode", "week", TARGET]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def time_split(df: pd.DataFrame, test_weeks: int):
    weeks = sorted(df["week"].unique())
    cutoff = weeks[-test_weeks]
    return df[df["week"] < cutoff], df[df["week"] >= cutoff]


def get_champion(client: MlflowClient, name: str):
    try:
        mv = client.get_model_version_by_alias(name, "champion")
        return mv, client.get_run(mv.run_id).data.metrics["mae"]
    except MlflowException:
        return None, None


def should_promote(new_mae: float, champion_mae: float | None, tolerance: float,
                   baseline_mae: float | None = None) -> bool:
    if baseline_mae is not None and new_mae >= baseline_mae:
        return False
    return champion_mae is None or new_mae <= champion_mae * (1 + tolerance)


def main() -> None:
    cfg = load_config()
    tc = cfg["training"]
    tc["params"].update({k: int(v) for k, v in (a.split("=") for a in sys.argv[1:])})
    features_path = Path(cfg["features"]["output_path"])
    df = pd.read_parquet(features_path)

    train, test = time_split(df, tc["test_weeks"])
    cols = [c for c in df.columns if c not in NON_FEATURES]

    mlflow.set_tracking_uri(tc["tracking_uri"])
    mlflow.set_experiment(tc["experiment_name"])
    client = MlflowClient()

    with mlflow.start_run() as run:
        mlflow.set_tags({"git_commit": git_commit(), "features_sha256": file_sha256(features_path)})
        mlflow.log_params(tc["params"])
        mlflow.log_params({"n_train": len(train), "n_test": len(test), "test_weeks": tc["test_weeks"]})

        model = RandomForestRegressor(**tc["params"], n_jobs=-1)
        model.fit(train[cols], train[TARGET])
        pred = model.predict(test[cols])

        mae = float(mean_absolute_error(test[TARGET], pred))
        rmse = float(np.sqrt(mean_squared_error(test[TARGET], pred)))
        baseline_mae = float(mean_absolute_error(test[TARGET], test["qty_lag_1"]))
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "baseline_mae": baseline_mae})

        mlflow.sklearn.log_model(model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        mlflow.log_dict({"features": cols}, "features.json")

        mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", tc["model_name"])

    name = tc["model_name"]
    champion, champion_mae = get_champion(client, name)
    if should_promote(mae, champion_mae, tc["promotion_tolerance"], baseline_mae):
        if champion is not None:
            client.set_model_version_tag(name, champion.version, "status", "archived")
        client.set_registered_model_alias(name, "champion", mv.version)
        client.set_model_version_tag(name, mv.version, "status", "champion")
        decision = "PROMU"
    else:
        client.set_model_version_tag(name, mv.version, "status", "rejected")
        if mae >= baseline_mae:
            decision = f"REJETÉ (ne bat pas la baseline : MAE {mae:.2f} >= {baseline_mae:.2f})"
        else:
            decision = f"REJETÉ (MAE champion = {champion_mae:.2f})"

    print(f"Version {mv.version} | MAE={mae:.2f} RMSE={rmse:.2f} baseline_MAE={baseline_mae:.2f} -> {decision}")


if __name__ == "__main__":
    main()
