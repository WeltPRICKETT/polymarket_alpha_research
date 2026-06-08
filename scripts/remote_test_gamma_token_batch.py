import csv
import json
import os

import polars as pl
import requests


def load_market_tokens() -> set[str]:
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
                        tokens.update(str(x) for x in json.loads(raw) if x)
                    except Exception:
                        pass
                for col in ("token1", "token2"):
                    if row.get(col):
                        tokens.add(str(row[col]))
    return tokens


def load_missing_sample(n: int = 5) -> list[str]:
    df = (
        pl.scan_csv(
            "data/orderFilled.csv",
            schema_overrides={"makerAssetId": pl.Utf8, "takerAssetId": pl.Utf8},
        )
        .select(["makerAssetId", "takerAssetId"])
        .unique()
        .collect()
    )
    trade_tokens = set(df["makerAssetId"].drop_nulls().to_list())
    trade_tokens.update(df["takerAssetId"].drop_nulls().to_list())
    trade_tokens.discard("0")
    missing = sorted(trade_tokens - load_market_tokens())
    return missing[:n]


def main() -> None:
    sample = load_missing_sample(5)
    print("sample", sample)
    session = requests.Session()
    for mode, value in [
        ("single", sample[0]),
        ("comma", ",".join(sample)),
        ("jsonish", json.dumps(sample)),
    ]:
        for closed in ("true", "false"):
            resp = session.get(
                "https://gamma-api.polymarket.com/markets",
                params={"clob_token_ids": value, "closed": closed, "limit": 10},
                timeout=30,
            )
            print("MODE", mode, "closed", closed, "status", resp.status_code, "url", resp.url)
            text = resp.text[:500].replace("\n", " ")
            print("TEXT", text)
            try:
                payload = resp.json()
                rows = payload if isinstance(payload, list) else payload.get("markets") or payload.get("data") or []
                print("ROWS", len(rows), "IDS", [r.get("id") for r in rows[:5]])
            except Exception as exc:
                print("JSON_ERR", exc)


if __name__ == "__main__":
    main()
