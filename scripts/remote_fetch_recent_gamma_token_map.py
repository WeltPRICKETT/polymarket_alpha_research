import csv
import json
import os
import time
from pathlib import Path

import requests


BASE_URL = "https://gamma-api.polymarket.com/markets/keyset"
OUT = Path("data/missing_markets.csv")
STATE = Path("data/recent_gamma_token_map_state.json")
LOG_EVERY = 25
LIMIT = int(os.environ.get("RECENT_GAMMA_LIMIT", "500"))
MIN_ID = int(os.environ.get("RECENT_GAMMA_MIN_ID", "1400000"))
MAX_ERRORS = int(os.environ.get("RECENT_GAMMA_MAX_ERRORS", "20"))


def load_seen_ids() -> set[str]:
    seen = set()
    for fname in ("data/markets.csv", str(OUT)):
        path = Path(fname)
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "id" not in (reader.fieldnames or []):
                continue
            for row in reader:
                if row.get("id"):
                    seen.add(str(row["id"]))
    return seen


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {"closed": {}, "active": {}}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True))


def append_rows(rows: list[dict], writer: csv.writer, seen: set[str]) -> int:
    wrote = 0
    for row in rows:
        market_id = str(row.get("id") or "")
        token_ids = row.get("clobTokenIds")
        if not market_id or not token_ids:
            continue
        if market_id in seen:
            continue
        seen.add(market_id)
        if isinstance(token_ids, str):
            token_ids_json = token_ids
        else:
            token_ids_json = json.dumps(token_ids, ensure_ascii=False)
        writer.writerow([market_id, token_ids_json])
        wrote += 1
    return wrote


def fetch_pass(closed: bool, state: dict, seen: set[str], writer: csv.writer) -> None:
    key = "closed" if closed else "active"
    cursor = state.get(key, {}).get("cursor")
    pages = int(state.get(key, {}).get("pages", 0))
    written = int(state.get(key, {}).get("written", 0))
    lowest_id = int(state.get(key, {}).get("lowest_id", 10**18))
    completed = bool(state.get(key, {}).get("completed", False))
    if completed:
        print(f"{key}: already completed")
        return

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://polymarket.com",
            "Referer": "https://polymarket.com/",
        }
    )
    errors = 0
    print(f"=== {key.upper()} recent descending pass; min_id={MIN_ID} ===")
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
            print(f"{key}: network {exc}; errors={errors}/{MAX_ERRORS}; sleep={sleep}s")
            if errors >= MAX_ERRORS:
                break
            time.sleep(sleep)
            continue
        if response.status_code != 200:
            errors += 1
            sleep = min(120, 10 * errors)
            print(
                f"{key}: status={response.status_code}; "
                f"errors={errors}/{MAX_ERRORS}; sleep={sleep}s"
            )
            if errors >= MAX_ERRORS:
                break
            time.sleep(sleep)
            continue

        errors = 0
        payload = response.json()
        markets = payload.get("markets", [])
        next_cursor = payload.get("next_cursor")
        if not markets:
            completed = True
            print(f"{key}: no more markets")
            break

        ids = [int(m["id"]) for m in markets if str(m.get("id", "")).isdigit()]
        if ids:
            lowest_id = min(lowest_id, min(ids))
        wrote = append_rows(markets, writer, seen)
        written += wrote
        pages += 1
        cursor = next_cursor
        state[key] = {
            "cursor": cursor,
            "pages": pages,
            "written": written,
            "lowest_id": lowest_id,
            "completed": completed,
        }
        save_state(state)

        if pages % LOG_EVERY == 0 or wrote:
            print(
                f"{key}: pages={pages} wrote={written} "
                f"batch_wrote={wrote} lowest_id={lowest_id}"
            )

        if lowest_id < MIN_ID:
            completed = True
            state[key]["completed"] = True
            save_state(state)
            print(f"{key}: reached min_id with lowest_id={lowest_id}")
            break
        if not next_cursor:
            completed = True
            state[key]["completed"] = True
            save_state(state)
            print(f"{key}: pagination completed")
            break
        time.sleep(0.15)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT.exists() or OUT.stat().st_size == 0
    seen = load_seen_ids()
    print("seen_ids", len(seen), "out", OUT, "min_id", MIN_ID)
    state = load_state()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["id", "clobTokenIds"])
        fetch_pass(True, state, seen, writer)
        fetch_pass(False, state, seen, writer)
    print("done", json.dumps(load_state(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
