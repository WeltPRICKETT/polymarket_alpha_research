import csv
import json
import os

import polars as pl


def load_tokens() -> set[str]:
    tokens = set()
    for fname in ("data/markets.csv", "data/missing_markets.csv"):
        if not os.path.exists(fname):
            continue
        with open(fname, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("clobTokenIds")
                if raw:
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, str):
                            parsed = json.loads(parsed)
                        for token_id in parsed:
                            if token_id:
                                tokens.add(str(token_id))
                    except Exception:
                        pass
                for col in ("token1", "token2"):
                    token_id = row.get(col)
                    if token_id:
                        tokens.add(str(token_id))
    return tokens


def main() -> None:
    tokens = load_tokens()
    print("source_tokens", len(tokens))
    df = pl.scan_csv(
        "data/orderFilled.csv",
        schema_overrides={"makerAssetId": pl.Utf8, "takerAssetId": pl.Utf8},
    ).select(
        [
            pl.from_epoch(pl.col("timestamp"), time_unit="s").dt.date().alias("date"),
            pl.when(pl.col("makerAssetId") != "0")
            .then(pl.col("makerAssetId"))
            .otherwise(pl.col("takerAssetId"))
            .alias("asset_id"),
        ]
    )
    out = (
        df.with_columns(pl.col("asset_id").is_in(tokens).alias("known"))
        .group_by("date")
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("asset_id").n_unique().alias("unique_tokens"),
                pl.col("known").sum().alias("known_rows"),
                pl.col("asset_id").filter(pl.col("known")).n_unique().alias("known_tokens"),
            ]
        )
        .with_columns(
            [
                (pl.col("known_rows") / pl.col("rows")).alias("row_coverage"),
                (pl.col("known_tokens") / pl.col("unique_tokens")).alias("token_coverage"),
            ]
        )
        .sort("date")
        .collect()
    )
    print(out)
    out.write_csv("data/token_coverage_by_day.csv")
    print("wrote data/token_coverage_by_day.csv")


if __name__ == "__main__":
    main()
