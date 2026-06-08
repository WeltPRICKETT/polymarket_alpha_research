import csv
import json
import os
import time

import polars as pl


def main() -> None:
    start = time.time()
    print("DIAG_START")

    market_tokens = set()
    market_rows = 0
    for fname in ("data/markets.csv", "data/missing_markets.csv"):
        if not os.path.exists(fname):
            continue
        with open(fname, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                market_rows += 1
                clob_token_ids = row.get("clobTokenIds")
                if clob_token_ids:
                    try:
                        for token_id in json.loads(clob_token_ids):
                            if token_id:
                                market_tokens.add(str(token_id))
                    except Exception:
                        pass
                for col in ("token1", "token2"):
                    token_id = row.get(col)
                    if token_id:
                        market_tokens.add(str(token_id))

    print(
        "market_rows",
        market_rows,
        "market_tokens",
        len(market_tokens),
        "load_sec",
        round(time.time() - start, 2),
    )

    df = (
        pl.scan_csv(
            "data/orderFilled.csv",
            schema_overrides={"makerAssetId": pl.Utf8, "takerAssetId": pl.Utf8},
        )
        .select(["makerAssetId", "takerAssetId"])
        .unique()
        .collect()
    )
    trade_asset_ids = set(df["makerAssetId"].drop_nulls().to_list()) | set(
        df["takerAssetId"].drop_nulls().to_list()
    )
    trade_asset_ids.discard("0")
    missing = trade_asset_ids - market_tokens
    coverage = 1 - len(missing) / max(len(trade_asset_ids), 1)

    print(
        "trade_asset_ids",
        len(trade_asset_ids),
        "missing_tokens",
        len(missing),
        "coverage",
        round(coverage, 6),
        "total_sec",
        round(time.time() - start, 2),
    )
    print("missing_sample", list(sorted(missing))[:20])


if __name__ == "__main__":
    main()
