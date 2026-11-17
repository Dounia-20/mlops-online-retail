import sys

import mlflow
from mlflow.tracking import MlflowClient

from src.data.quality import load_config


def main() -> None:
    tc = load_config()["training"]
    name = tc["model_name"]
    mlflow.set_tracking_uri(tc["tracking_uri"])
    client = MlflowClient()

    if len(sys.argv) != 2:
        print("Versions disponibles :")
        for mv in sorted(client.search_model_versions(f"name='{name}'"), key=lambda m: int(m.version)):
            mae = client.get_run(mv.run_id).data.metrics.get("mae")
            print(f"  v{mv.version} | statut={mv.tags.get('status')} | MAE={mae:.2f}")
        print("Usage : python -m src.training.rollback <numero_version>")
        return

    target = sys.argv[1]
    target_version = client.get_model_version(name, target)
    if target_version.tags.get("status") == "rejected":
        raise SystemExit(f"Retour arrière refusé : la version {target} a été rejetée par la règle de mise en service.")
    try:
        current = client.get_model_version_by_alias(name, "champion")
        client.set_model_version_tag(name, current.version, "status", "archived")
    except Exception:
        current = None

    client.set_registered_model_alias(name, "champion", target)
    client.set_model_version_tag(name, target, "status", "champion")
    avant = f"v{current.version}" if current else "aucun"
    print(f"Retour arrière effectué : champion {avant} -> v{target}")


if __name__ == "__main__":
    main()
