import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from prefect import flow, get_run_logger, task

from src.data import ingest
from src.data.quality import DataQualityError, load_config, validate_batch_file
from src.features import build_features
from src.training import train


def alert(message: str) -> None:
    path = Path(load_config()["orchestration"]["alerts_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().isoformat(timespec='seconds')} ALERTE {message}"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def record_run(flow_run, status: str) -> None:
    from src.api import db
    try:
        duration = (datetime.now(timezone.utc) - flow_run.start_time).total_seconds()
    except Exception:
        duration = None
    db.log_pipeline_run(flow_run.name, status, duration)


def on_flow_completion(flow, flow_run, state) -> None:
    record_run(flow_run, "success")


def on_flow_failure(flow, flow_run, state) -> None:
    record_run(flow_run, "failure")
    alert(f"Pipeline '{flow_run.name}' en échec : {state.message}")


@task(retries=2, retry_delay_seconds=5)
def ingest_task() -> None:
    ingest.main()


@task
def quality_task() -> int:
    cfg = load_config()
    paths = sorted(Path(cfg["data"]["batches_dir"]).glob("batch_*.parquet"))
    if not paths:
        raise FileNotFoundError("Aucun lot trouvé : lancer l'ingestion d'abord")
    errors = []
    for p in paths:
        try:
            validate_batch_file(p, cfg)
        except DataQualityError as e:
            errors.append(str(e))
    if errors:
        raise DataQualityError(" || ".join(errors))
    return len(paths)


@task
def features_task() -> None:
    build_features.main()


@task
def train_task() -> None:
    train.main()


@task
def reload_api_task() -> None:
    logger = get_run_logger()
    url = load_config()["orchestration"]["api_url"]
    try:
        r = httpx.post(f"{url}/reload", timeout=10)
        r.raise_for_status()
        logger.info(f"API rechargée : {r.json()}")
    except Exception as e:
        logger.warning(f"API non rechargée ({e})")


@flow(name="online-retail-pipeline", on_failure=[on_flow_failure], on_completion=[on_flow_completion])
def pipeline(skip_ingest: bool = False) -> None:
    if not skip_ingest:
        ingest_task()
    quality_task()
    features_task()
    train_task()
    reload_api_task()


if __name__ == "__main__":
    skip = "--skip-ingest" in sys.argv
    serve = "--serve" in sys.argv
    cron = sys.argv[sys.argv.index("--cron") + 1] if "--cron" in sys.argv else None
    sys.argv = sys.argv[:1]
    if serve:
        pipeline.serve(name="pipeline-mensuel", cron=cron or load_config()["orchestration"]["schedule_cron"])
    else:
        pipeline(skip_ingest=skip)


