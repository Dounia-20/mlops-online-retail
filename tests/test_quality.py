import pandas as pd
import pytest

from src.data.quality import check_batch, load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def valid_df():
    n = 200
    return pd.DataFrame({
        "InvoiceNo": [str(500000 + i) for i in range(n)],
        "StockCode": ["85123A"] * n,
        "Description": pd.array(["WHITE HANGING HEART"] * n, dtype="string"),
        "Quantity": [6] * n,
        "InvoiceDate": pd.date_range("2011-01-01", periods=n, freq="h"),
        "UnitPrice": [2.55] * n,
        "CustomerID": [17850.0] * n,
        "Country": ["United Kingdom"] * n,
    })


def test_valid_batch_passes(valid_df, cfg):
    assert check_batch(valid_df, cfg) == []


def test_missing_column_is_blocked(valid_df, cfg):
    errors = check_batch(valid_df.drop(columns=["Quantity"]), cfg)
    assert any("Colonnes manquantes" in e for e in errors)


def test_too_few_rows_is_blocked(valid_df, cfg):
    errors = check_batch(valid_df.head(10), cfg)
    assert any("Trop peu de lignes" in e for e in errors)


def test_null_quantity_is_blocked(valid_df, cfg):
    valid_df.loc[0, "Quantity"] = None
    errors = check_batch(valid_df, cfg)
    assert any("valeurs nulles dans Quantity" in e for e in errors)


def test_wrong_type_is_blocked(valid_df, cfg):
    valid_df["UnitPrice"] = "abc"
    errors = check_batch(valid_df, cfg)
    assert any("UnitPrice n'est pas numérique" in e for e in errors)
