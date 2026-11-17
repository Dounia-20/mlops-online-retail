import sys
from pathlib import Path

import pandas as pd
import yaml


class DataQualityError(Exception):
    pass


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_batch(df: pd.DataFrame, cfg: dict) -> list[str]:
    q = cfg["quality"]
    errors = []

    missing = [c for c in q["required_columns"] if c not in df.columns]
    if missing:
        return [f"Colonnes manquantes : {missing}"]

    if len(df) < q["min_rows_per_batch"]:
        errors.append(f"Trop peu de lignes : {len(df)} < {q['min_rows_per_batch']}")

    for col in ["InvoiceNo", "StockCode", "InvoiceDate", "Quantity", "UnitPrice"]:
        n = int(df[col].isna().sum())
        if n:
            errors.append(f"{n} valeurs nulles dans {col}")

    ratio = df["Description"].isna().mean()
    if ratio > q["max_null_ratio_description"]:
        errors.append(f"Description nulle à {ratio:.2%} (max {q['max_null_ratio_description']:.2%})")

    if not pd.api.types.is_numeric_dtype(df["Quantity"]):
        errors.append("Quantity n'est pas numérique")
    if not pd.api.types.is_numeric_dtype(df["UnitPrice"]):
        errors.append("UnitPrice n'est pas numérique")
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        errors.append("InvoiceDate n'est pas une date")

    return errors


def validate_batch_file(path: Path, cfg: dict) -> None:
    errors = check_batch(pd.read_parquet(path), cfg)
    if errors:
        raise DataQualityError(f"{path.name} : " + " | ".join(errors))


def main() -> None:
    cfg = load_config()
    failed = 0
    for p in sorted(Path(cfg["data"]["batches_dir"]).glob("batch_*.parquet")):
        try:
            validate_batch_file(p, cfg)
            print(f"OK    {p.name}")
        except DataQualityError as e:
            failed += 1
            print(f"ECHEC {e}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
