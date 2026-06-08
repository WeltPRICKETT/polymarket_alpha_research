import csv

import polars as pl


WINDOW = "processed/trades_window_20260525_20260527_exclusive.csv"
MARKETS = "data/markets.csv"
MISSING = "data/missing_markets.csv"


def read_ids(path: str) -> set[str]:
    ids = set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id"):
                    ids.add(str(row["id"]))
    except FileNotFoundError:
        pass
    return ids


window_markets = set(
    pl.scan_csv(WINDOW, schema_overrides={"market_id": pl.Utf8})
    .select(pl.col("market_id").unique())
    .collect()["market_id"]
    .to_list()
)
full_ids = read_ids(MARKETS)
missing_ids = read_ids(MISSING)

full = window_markets & full_ids
token_only = (window_markets & missing_ids) - full_ids
unknown = window_markets - full_ids - missing_ids

print("window_markets", len(window_markets))
print("full_metadata_markets", len(full))
print("token_only_missing_markets", len(token_only))
print("unknown_markets", len(unknown))
print("token_only_sample", sorted(token_only)[:20])
print("unknown_sample", sorted(unknown)[:20])
