import csv
import json
import os
import time
from pathlib import Path

import requests


BASE_URL = "https://gamma-api.polymarket.com/markets/keyset"
OUT = Path(os.environ.get("RECENT_FULL_OUT", "data/recent_markets_full.csv"))
STATE = Path(os.environ.get("RECENT_FULL_STATE", "data/recent_markets_full_state.json"))
LIMIT = int(os.environ.get("RECENT_FULL_LIMIT", "500"))
MIN_ID = int(os.environ.get("RECENT_FULL_MIN_ID", "2319000"))
MAX_ERRORS = int(os.environ.get("RECENT_FULL_MAX_ERRORS", "20"))


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {"closed": {}, "active": {}}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True))


def flatten(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def fetch_pass(closed: bool, state: dict, writer: csv.writer, columns: list[str] | None) -> list[str] | None:
    key = "closed" if closed else "active"
    part = state.get(key, {})
    cursor = part.get("cursor")
    pages = int(part.get("pages", 0))
    written = int(part.get("written", 0))
    lowest_id = int(part.get("lowest_id", 10**18))
    completed = bool(part.get("completed", False))
    if completed:
        print(f"{key}: already completed")
        return columns

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 Chrome/120 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://polymarket.com",
            "Referer": "https://polymarket.com/",
        }
    )
    errors = 0
    print(f"=== {key.upper()} full descending; min_id={MIN_ID} ===", flush=True)
    while True:
        params = {
            "closed": "true" if closed else "false",
            "limit": LIMIT,
            "order": "id",
            "ascending": "false",
        }
        if cursor:
            params["after_cursor"] = cursor
        try:
            response = session.get(BASE_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            errors += 1
            sleep = min(120, 10 * errors)
            print(f"{key}: network {exc}; errors={errors}/{MAX_ERRORS}; sleep={sleep}s", flush=True)
            if errors >= MAX_ERRORS:
                break
            time.sleep(sleep)
            continue
        if response.status_code != 200:
            errors += 1
            sleep = min(120, 10 * errors)
            print(f"{key}: status={response.status_code}; errors={errors}/{MAX_ERRORS}; sleep={sleep}s", flush=True)
            if errors >= MAX_ERRORS:
                break
            time.sleep(sleep)
            continue

        errors = 0
        payload = response.json()
        markets = payload.get("markets", [])
        cursor = payload.get("next_cursor")
        if not markets:
            completed = True
            break
        if columns is None:
            columns = list(markets[0].keys())
            writer.writerow(columns)

        ids = [int(m["id"]) for m in markets if str(m.get("id", "")).isdigit()]
        if ids:
            lowest_id = min(lowest_id, min(ids))
        for market in markets:
            writer.writerow([flatten(market.get(col)) for col in columns])
        pages += 1
        written += len(markets)
        state[key] = {
            "cursor": cursor,
            "pages": pages,
            "written": written,
            "lowest_id": lowest_id,
            "completed": completed,
        }
        save_state(state)
        if pages % 10 == 0:
            print(f"{key}: pages={pages} written={written} lowest_id={lowest_id}", flush=True)
        if lowest_id < MIN_ID or not cursor:
            state[key]["completed"] = True
            save_state(state)
            print(f"{key}: stopping lowest_id={lowest_id} cursor={bool(cursor)}", flush=True)
            break
        time.sleep(0.12)
    return columns


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_new = not OUT.exists() or OUT.stat().st_size == 0
    columns = None
    if not write_new:
        with OUT.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
            columns = header
    state = load_state()
    mode = "a" if OUT.exists() else "w"
    with OUT.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        columns = fetch_pass(True, state, writer, columns)
        columns = fetch_pass(False, state, writer, columns)
    print("done", json.dumps(load_state(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
