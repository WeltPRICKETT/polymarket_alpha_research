import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone


def load_known_tokens() -> set[str]:
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
                        for token in parsed:
                            if token:
                                tokens.add(str(token))
                    except Exception:
                        pass
                for col in ("token1", "token2"):
                    token = row.get(col)
                    if token:
                        tokens.add(str(token))
    return tokens


def main() -> None:
    start = time.time()
    known = load_known_tokens()
    print("known_tokens", len(known), "load_sec", round(time.time() - start, 2), flush=True)

    stats = defaultdict(lambda: {"rows": 0, "known_rows": 0, "tokens": set(), "known_tokens": set()})
    total_rows = 0
    with open("data/orderFilled.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if total_rows % 5_000_000 == 0:
                print("rows", total_rows, "sec", round(time.time() - start, 1), flush=True)
            asset = row["makerAssetId"] if row["makerAssetId"] != "0" else row["takerAssetId"]
            if not asset or asset == "0":
                continue
            day = datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc).date().isoformat()
            bucket = stats[day]
            bucket["rows"] += 1
            bucket["tokens"].add(asset)
            if asset in known:
                bucket["known_rows"] += 1
                bucket["known_tokens"].add(asset)

    out = "data/token_coverage_by_day_stream.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "date",
                "rows",
                "unique_tokens",
                "known_rows",
                "known_tokens",
                "row_coverage",
                "token_coverage",
            ]
        )
        for day in sorted(stats):
            bucket = stats[day]
            rows = bucket["rows"]
            unique_tokens = len(bucket["tokens"])
            known_rows = bucket["known_rows"]
            known_tokens = len(bucket["known_tokens"])
            writer.writerow(
                [
                    day,
                    rows,
                    unique_tokens,
                    known_rows,
                    known_tokens,
                    known_rows / rows if rows else 0,
                    known_tokens / unique_tokens if unique_tokens else 0,
                ]
            )
    print("done", total_rows, "out", out, "sec", round(time.time() - start, 2), flush=True)


if __name__ == "__main__":
    main()
