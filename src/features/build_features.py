from pathlib import Path

import pandas as pd

from src.data.quality import load_config, validate_batch_file


def load_validated_batches(cfg: dict) -> pd.DataFrame:
    paths = sorted(Path(cfg["data"]["batches_dir"]).glob("batch_*.parquet"))
    if not paths:
        raise FileNotFoundError("Aucun lot trouvé : lancer d'abord src/data/ingest.py")
    for p in paths:
        validate_batch_file(p, cfg)
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    return df[df["StockCode"].str.match(r"^\d{5}")]


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fc = cfg["features"]
    df = clean(df).copy()
    df["week"] = df["InvoiceDate"].dt.to_period("W").dt.start_time

    top = df.groupby("StockCode")["Quantity"].sum().nlargest(fc["top_n_products"]).index
    df = df[df["StockCode"].isin(top)]

    weekly = df.groupby(["StockCode", "week"]).agg(
        quantity=("Quantity", "sum"), price=("UnitPrice", "mean")
    )

    weeks = pd.date_range(df["week"].min(), df["week"].max(), freq="W-MON")
    grid = pd.MultiIndex.from_product([top, weeks], names=["StockCode", "week"])
    weekly = weekly.reindex(grid)
    weekly["quantity"] = weekly["quantity"].fillna(0)
    weekly["price"] = weekly.groupby(level=0)["price"].transform(lambda s: s.ffill().bfill())
    weekly = weekly.reset_index().sort_values(["StockCode", "week"])

    g = weekly.groupby("StockCode")
    for lag in fc["lags"]:
        weekly[f"qty_lag_{lag}"] = g["quantity"].shift(lag)
    weekly["qty_roll_mean"] = g["quantity"].transform(
        lambda s: s.shift(1).rolling(fc["rolling_window"]).mean()
    )
    weekly["price_lag_1"] = g["price"].shift(1)
    weekly["week_of_year"] = weekly["week"].dt.isocalendar().week.astype(int)
    weekly["month"] = weekly["week"].dt.month

    weekly = weekly.drop(columns=["price"]).dropna().reset_index(drop=True)
    return weekly


def main() -> None:
    cfg = load_config()
    feats = build_features(load_validated_batches(cfg), cfg)
    out = Path(cfg["features"]["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"Features écrites : {out} {feats.shape}")
    print(feats.head())


if __name__ == "__main__":
    main()
