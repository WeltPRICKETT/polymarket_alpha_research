import csv
import json
import os
import time

import polars as pl


POLY_ROOT_MARKETS = ("data/markets.csv", "data/missing_markets.csv")
SII_MARKETS = (
    "/home/normal/polymarket_alpha_gpu/polymarket_alpha_rich/"
    "data/external/sii_polymarket_data/markets.parquet"
)


def load_poly_tokens() -> set[str]:
    tokens = set()
    for fname in POLY_ROOT_MARKETS:
        if not os.path.exists(fname):
            continue
        with open(fname, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("clobTokenIds")
                if raw:
                    try:
                        for token_id in json.loads(raw):
                            if token_id:
                                tokens.add(str(token_id))
                    except Exception:
                        pass
                for col in ("token1", "token2"):
                    token_id = row.get(col)
                    if token_id:
                        tokens.add(str(token_id))
    return tokens


def load_sii_tokens() -> set[str]:
    if not os.path.exists(SII_MARKETS):
        return set()
    df = pl.scan_parquet(SII_MARKETS).select(["token1", "token2"]).collect()
    tokens = set(df["token1"].drop_nulls().cast(pl.Utf8).to_list())
    tokens.update(df["token2"].drop_nulls().cast(pl.Utf8).to_list())
    tokens.discard("")
    return tokens


def load_trade_tokens() -> set[str]:
    df = (
        pl.scan_csv(
            "data/orderFilled.csv",
            schema_overrides={"makerAssetId": pl.Utf8, "takerAssetId": pl.Utf8},
        )
        .select(["makerAssetId", "takerAssetId"])
        .unique()
        .collect()
    )
    tokens = set(df["makerAssetId"].drop_nulls().to_list())
    tokens.update(df["takerAssetId"].drop_nulls().to_list())
    tokens.discard("0")
    tokens.discard("")
    return tokens


def report(name: str, trade_tokens: set[str], source_tokens: set[str]) -> None:
    missing = trade_tokens - source_tokens
    coverage = 1 - len(missing) / max(len(trade_tokens), 1)
    print(
        name,
        "source_tokens",
        len(source_tokens),
        "missing",
        len(missing),
        "coverage",
        round(coverage, 6),
    )
    print(name, "missing_sample", list(sorted(missing))[:10])


def main() -> None:
    start = time.time()
    trade_tokens = load_trade_tokens()
    print("trade_tokens", len(trade_tokens), "sec", round(time.time() - start, 2))
    poly_tokens = load_poly_tokens()
    print("poly_tokens_loaded", len(poly_tokens), "sec", round(time.time() - start, 2))
    sii_tokens = load_sii_tokens()
    print("sii_tokens_loaded", len(sii_tokens), "sec", round(time.time() - start, 2))
    report("poly_only", trade_tokens, poly_tokens)
    report("sii_only", trade_tokens, sii_tokens)
    report("combined", trade_tokens, poly_tokens | sii_tokens)


if __name__ == "__main__":
    main()
