from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_raw(raw_path: str) -> pd.DataFrame:
    df = pd.read_excel(raw_path)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].astype("string")
    return df


def split_into_batches(df: pd.DataFrame, batches_dir: str) -> list[Path]:
    out_dir = Path(batches_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for period, chunk in df.groupby(df["InvoiceDate"].dt.to_period("M")):
        p = out_dir / f"batch_{period}.parquet"
        chunk.to_parquet(p, index=False)
        paths.append(p)
    return paths


def main() -> None:
    cfg = load_config()
    df = read_raw(cfg["data"]["raw_path"])

    parquet_path = Path(cfg["data"]["parquet_path"])
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    print(f"Parquet écrit : {parquet_path} ({len(df)} lignes)")

    paths = split_into_batches(df, cfg["data"]["batches_dir"])
    print(f"{len(paths)} lots écrits dans {cfg['data']['batches_dir']}")


if __name__ == "__main__":
    main()
