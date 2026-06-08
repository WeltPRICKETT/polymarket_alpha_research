import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import requests


WINDOW = os.environ.get("WINDOW_TRADES_CSV", "processed/trades_window_20260525_20260527_exclusive.csv")
MARKETS = os.environ.get("LEAN_MARKETS_CSV", "data/markets.csv")
MISSING = os.environ.get("MISSING_MARKETS_CSV", "data/missing_markets.csv")
OUT = Path(os.environ.get("WINDOW_FULL_OUT", "data/window_missing_markets_full.csv"))
TARGET_MARKET_IDS_CSV = os.environ.get("TARGET_MARKET_IDS_CSV", "")
MAX_WORKERS = int(os.environ.get("WINDOW_FULL_WORKERS", "8"))
MAX_RETRIES = int(os.environ.get("WINDOW_FULL_RETRIES", "4"))
REQ_TIMEOUT = int(os.environ.get("WINDOW_FULL_TIMEOUT", "8"))
LOG_EVERY = int(os.environ.get("WINDOW_FULL_LOG_EVERY", "25"))


def parse_tokens(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        return [str(x) for x in parsed if x]
    except Exception:
        return []


def read_full_ids() -> set[str]:
    ids = set()
    for fname in (MARKETS, str(OUT)):
        if not os.path.exists(fname):
            continue
        with open(fname, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "id" not in (reader.fieldnames or []):
                continue
            for row in reader:
                if row.get("id"):
                    ids.add(str(row["id"]))
    return ids


def read_target_ids() -> set[str] | None:
    if not TARGET_MARKET_IDS_CSV:
        return None
    ids = set()
    with open(TARGET_MARKET_IDS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        id_col = "market_id" if "market_id" in fieldnames else "id" if "id" in fieldnames else None
        if id_col is None:
            raise ValueError(f"{TARGET_MARKET_IDS_CSV} must contain market_id or id")
        for row in reader:
            mid = str(row.get(id_col) or "").strip()
            if mid:
                ids.add(mid)
    return ids


def load_targets() -> list[tuple[str, list[str]]]:
    target_ids = read_target_ids()
    if target_ids is None:
        window_ids = set(
            pl.scan_csv(WINDOW, schema_overrides={"market_id": pl.Utf8})
            .select(pl.col("market_id").unique())
            .collect()["market_id"]
            .to_list()
        )
    else:
        window_ids = target_ids
    full_ids = read_full_ids()
    targets = []
    with open(MISSING, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = str(row.get("id") or "")
            if mid not in window_ids or mid in full_ids:
                continue
            tokens = parse_tokens(row.get("clobTokenIds") or "")
            if tokens:
                targets.append((mid, tokens))
    return targets


def fetch_one(item: tuple[str, list[str]]) -> tuple[str, dict | None, str]:
    mid, tokens = item
    session = requests.Session()
    for token in tokens:
        for closed in ("true", "false"):
            for attempt in range(MAX_RETRIES):
                try:
                    response = session.get(
                        "https://gamma-api.polymarket.com/markets",
                        params={"clob_token_ids": token, "closed": closed, "limit": 3},
                        timeout=REQ_TIMEOUT,
                    )
                except requests.RequestException as exc:
                    time.sleep(1.5 * (attempt + 1))
                    last = f"network:{exc}"
                    continue
                if response.status_code != 200:
                    last = f"status:{response.status_code}"
                    time.sleep(1.5 * (attempt + 1))
                    continue
                payload = response.json()
                rows = payload if isinstance(payload, list) else payload.get("markets") or payload.get("data") or []
                for row in rows:
                    if str(row.get("id")) == mid:
                        return mid, row, "ok"
                if rows:
                    last = "rows_no_id_match"
                else:
                    last = "empty"
                break
    return mid, None, last if "last" in locals() else "not_found"


def flatten(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def main() -> None:
    targets = load_targets()
    print("targets", len(targets), "workers", MAX_WORKERS, "target_csv", TARGET_MARKET_IDS_CSV or "window", flush=True)
    rows = []
    failures = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fetch_one, target) for target in targets]
        for i, fut in enumerate(as_completed(futures), 1):
            mid, row, status = fut.result()
            if row:
                rows.append(row)
            else:
                failures.append((mid, status))
            if i % LOG_EVERY == 0:
                print("done", i, "ok", len(rows), "fail", len(failures), "sec", round(time.time() - start, 1), flush=True)

    if rows:
        columns = sorted({key for row in rows for key in row.keys()})
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([flatten(row.get(col)) for col in columns])
    fail_path = OUT.with_suffix(".failures.csv")
    with fail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "status"])
        writer.writerows(failures)
    print("wrote", len(rows), OUT, "failures", len(failures), fail_path, flush=True)


if __name__ == "__main__":
    main()
