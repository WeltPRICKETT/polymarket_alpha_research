#!/usr/bin/env python3
"""Build frozen-validator daily cache CSVs from an isolated late-window slice.

This adapter intentionally does not mutate the canonical SII parquet store.  It
converts the output of ``remote_process_live_window.py`` into the
``daily_top_wallet_trades_YYYYMMDD_<policy>.csv`` files consumed by
``research/strategy_track/validate_frozen_clusteraware_strategy.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_TRADE_COLUMNS = {
    "timestamp",
    "market_id",
    "maker",
    "taker",
    "nonusdc_side",
    "maker_direction",
    "taker_direction",
    "price",
    "usd_amount",
}

CACHE_COLUMNS = [
    "fold",
    "address",
    "market_id",
    "condition_id",
    "event_id",
    "price",
    "usd_amount",
    "direction",
    "nonusdc_side",
    "timestamp",
    "answer1",
    "answer2",
    "event_title",
    "market_volume",
    "end_date",
    "resolution",
    "delta",
    "signal_side",
    "dollar_delta",
    "category",
]

REQUIRED_METADATA_FOR_SIGNALS = ["answer1", "answer2", "event_id", "event_title"]
METADATA_COLUMNS = [
    "market_id",
    "condition_id",
    "event_id",
    "event_title",
    "answer1",
    "answer2",
    "market_volume",
    "end_date",
    "resolution",
]

REGIME = "live_plus_static_event_shape"
THRESHOLD = 0.003
RECENCY_DECAY = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create isolated late-window daily_top_wallet_trades caches for the "
            "frozen cluster-aware validator."
        )
    )
    parser.add_argument("--trades-csv", required=True, help="remote_process_live_window.py output CSV.")
    parser.add_argument(
        "--metadata-csv",
        action="append",
        required=True,
        help="Full market metadata CSV. Repeat for data/markets.csv and window_missing_markets_full.csv.",
    )
    parser.add_argument(
        "--wallet-source",
        required=True,
        help=(
            "Either an active top-wallet snapshot with fold,address,score columns, "
            "or a predictions/selected-wallet CSV with regime,fold,address,score."
        ),
    )
    parser.add_argument(
        "--wallet-source-mode",
        choices=["auto", "active", "predictions"],
        default="auto",
        help="How to interpret --wallet-source.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for daily cache outputs.")
    parser.add_argument("--start-date", default="2026-05-25", help="Inclusive UTC date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2026-05-27", help="Exclusive UTC date, YYYY-MM-DD.")
    parser.add_argument("--policy", choices=["ensemble", "recency_decay", "latest", "all"], default="ensemble")
    parser.add_argument("--regime", default=REGIME)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument(
        "--active-folds",
        default=None,
        help="Comma-separated folds to use when --wallet-source is predictions-like and no fold window file is given.",
    )
    parser.add_argument(
        "--fold-windows-csv",
        default=None,
        help="Optional CSV with fold,future_start,future_end epoch seconds for selecting active folds per day.",
    )
    parser.add_argument(
        "--resolutions-csv",
        default=None,
        help="Optional CSV with market_id or condition_id plus resolution/winning_outcome.",
    )
    parser.add_argument(
        "--schema-reference-csv",
        default=None,
        help="Optional existing daily_top_wallet_trades CSV used to verify column parity.",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help="Write cache files even when emitted top-wallet rows lack answer/event metadata.",
    )
    return parser.parse_args()


def suffix_for(policy: str) -> str:
    return "" if policy == "all" else f"_{policy}"


def parse_jsonish_list(value: object) -> list[object]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    for _ in range(2):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            break
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, str):
            text = parsed
            continue
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [part.strip().strip("'\"") for part in text.split(",") if part.strip()]


def first_present(row: pd.Series, names: Iterable[str]) -> object:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]).strip() != "":
            return row[name]
    return None


def extract_event_id(row: pd.Series) -> object:
    direct = first_present(row, ["event_id", "eventId", "eventID"])
    if direct is not None:
        return direct
    events = parse_jsonish_list(first_present(row, ["events", "event"]) or "")
    if events and isinstance(events[0], dict):
        return first_present(pd.Series(events[0]), ["id", "event_id", "eventId"])
    return None


def extract_event_title(row: pd.Series) -> object:
    direct = first_present(row, ["event_title", "eventTitle", "title", "slug"])
    if direct is not None:
        return direct
    events = parse_jsonish_list(first_present(row, ["events", "event"]) or "")
    if events and isinstance(events[0], dict):
        return first_present(pd.Series(events[0]), ["title", "event_title", "eventTitle", "slug"])
    return first_present(row, ["question", "description"])


def decisive_resolution(row: pd.Series, answer1: object, answer2: object) -> object:
    direct = first_present(row, ["resolution", "winning_outcome", "winner", "winningOutcome"])
    if direct is not None:
        return direct
    prices = parse_jsonish_list(first_present(row, ["outcome_prices", "outcomePrices", "outcomePricesJson"]) or "")
    if len(prices) < 2:
        return None
    try:
        vals = [float(x) for x in prices[:2]]
    except (TypeError, ValueError):
        return None
    if vals[0] >= 0.99 and vals[1] <= 0.01:
        return answer1
    if vals[1] >= 0.99 and vals[0] <= 0.01:
        return answer2
    return None


def normalize_metadata_row(row: pd.Series) -> dict[str, object]:
    outcomes = parse_jsonish_list(first_present(row, ["outcomes", "outcomes_json", "outcomesJson"]) or "")
    answer1 = first_present(row, ["answer1", "outcome1"])
    answer2 = first_present(row, ["answer2", "outcome2"])
    if answer1 is None and len(outcomes) >= 1:
        answer1 = outcomes[0]
    if answer2 is None and len(outcomes) >= 2:
        answer2 = outcomes[1]

    market_id = first_present(row, ["market_id", "id"])
    condition_id = first_present(row, ["condition_id", "conditionId", "conditionID"])
    resolution = decisive_resolution(row, answer1, answer2)
    return {
        "market_id": str(market_id) if market_id is not None else "",
        "condition_id": str(condition_id) if condition_id is not None else "",
        "event_id": extract_event_id(row),
        "event_title": extract_event_title(row),
        "answer1": answer1,
        "answer2": answer2,
        "market_volume": first_present(row, ["market_volume", "volume", "volumeNum", "liquidity"]),
        "end_date": first_present(row, ["end_date", "endDate", "endDateIso", "endDateISO"]),
        "resolution": resolution,
    }


def combine_first_nonempty(group: pd.DataFrame) -> pd.Series:
    out = {}
    for col in group.columns:
        if col == "market_id":
            out[col] = group[col].iloc[0]
            continue
        vals = group[col]
        nonempty = vals[vals.notna() & vals.astype(str).str.strip().ne("")]
        out[col] = nonempty.iloc[0] if not nonempty.empty else np.nan
    return pd.Series(out)


def collect_market_ids(trades_csv: Path, start_ts: int, end_ts: int, chunksize: int) -> tuple[set[str], dict[str, int]]:
    market_ids: set[str] = set()
    stats = {"input_rows_in_window": 0}
    for chunk in pd.read_csv(trades_csv, chunksize=chunksize, dtype=str):
        missing = REQUIRED_TRADE_COLUMNS - set(chunk.columns)
        if missing:
            raise ValueError(f"{trades_csv} missing required columns: {sorted(missing)}")
        ts = normalize_timestamp(chunk["timestamp"])
        mask = ts.ge(start_ts) & ts.lt(end_ts)
        stats["input_rows_in_window"] += int(mask.sum())
        market_ids.update(chunk.loc[mask, "market_id"].dropna().astype(str).tolist())
    return market_ids, stats


def load_metadata(paths: list[Path], needed_market_ids: set[str], chunksize: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for chunk in pd.read_csv(path, chunksize=chunksize, dtype=str, low_memory=False, on_bad_lines="skip"):
            id_col = "market_id" if "market_id" in chunk.columns else "id" if "id" in chunk.columns else None
            if id_col is None:
                continue
            sub = chunk[chunk[id_col].astype(str).isin(needed_market_ids)]
            if sub.empty:
                continue
            norm_rows = [normalize_metadata_row(row) for _, row in sub.iterrows()]
            rows.append(pd.DataFrame(norm_rows))
    if not rows:
        return pd.DataFrame(columns=METADATA_COLUMNS)
    meta = pd.concat(rows, ignore_index=True)
    meta = meta[meta["market_id"].astype(str).str.len().gt(0)].copy()
    meta["market_id"] = meta["market_id"].astype(str)
    return pd.DataFrame([combine_first_nonempty(group) for _, group in meta.groupby("market_id", sort=False)]).reset_index(drop=True)


def load_resolutions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    key = "market_id" if "market_id" in df.columns else "condition_id" if "condition_id" in df.columns else None
    val = next((c for c in ["resolution", "winning_outcome", "winner", "winningOutcome"] if c in df.columns), None)
    if key is None or val is None:
        raise ValueError(f"{path} must contain market_id or condition_id plus a resolution column")
    return df[[key, val]].rename(columns={key: "_resolution_key", val: "resolution_from_file"})


def merge_resolutions(meta: pd.DataFrame, resolutions: pd.DataFrame | None) -> pd.DataFrame:
    if resolutions is None or resolutions.empty:
        return meta
    out = meta.copy()
    if "resolution" not in out.columns:
        out["resolution"] = pd.NA
    out["resolution"] = out["resolution"].astype("object")
    resolutions = resolutions.copy()
    resolutions["_resolution_key"] = resolutions["_resolution_key"].astype(str)
    resolutions["resolution_from_file"] = resolutions["resolution_from_file"].astype("object")
    for key in ["market_id", "condition_id"]:
        if key not in out.columns:
            continue
        out[key] = out[key].where(out[key].notna(), "").astype(str)
        out = out.merge(resolutions, left_on=key, right_on="_resolution_key", how="left")
        fill = out["resolution"].isna() | out["resolution"].astype(str).str.strip().eq("")
        out.loc[fill, "resolution"] = out.loc[fill, "resolution_from_file"]
        out = out.drop(columns=["_resolution_key", "resolution_from_file"])
    return out


def normalize_timestamp(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    missing = numeric.isna()
    if missing.any():
        parsed = pd.to_datetime(series[missing], errors="coerce", utc=True)
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        numeric.loc[missing] = (parsed - epoch).dt.total_seconds().where(parsed.notna(), np.nan)
    return numeric.astype("Int64")


def date_epoch(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp())


def active_fold_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def policy_wallets_from_predictions(pred: pd.DataFrame, active_folds: list[int], policy: str, regime: str, threshold: float) -> pd.DataFrame:
    if not active_folds:
        return pd.DataFrame(columns=["fold", "address", "score"])
    rdf = pred[pred["regime"].astype(str).eq(regime)].copy() if "regime" in pred.columns else pred.copy()
    rdf["fold"] = pd.to_numeric(rdf["fold"], errors="coerce").astype("Int64")
    rdf["score"] = pd.to_numeric(rdf["score"], errors="coerce")
    rdf = rdf[rdf["fold"].isin(active_folds)].copy()
    if rdf.empty:
        return pd.DataFrame(columns=["fold", "address", "score"])
    rows = []
    for fold in active_folds:
        fdf = rdf[rdf["fold"].eq(fold)].sort_values("score", ascending=False).copy()
        if fdf.empty:
            continue
        n = max(1, int(len(fdf) * threshold))
        top = fdf.head(n).copy()
        top["rank_in_fold"] = np.arange(1, len(top) + 1)
        top["fold_top_n"] = n
        rows.append(top)
    top_rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if top_rows.empty:
        return pd.DataFrame(columns=["fold", "address", "score"])
    if policy == "all":
        return top_rows[["fold", "address", "score"]].copy()

    latest_fold = int(pd.to_numeric(top_rows["fold"], errors="coerce").dropna().max())
    latest_n = int(top_rows.loc[top_rows["fold"].eq(latest_fold), "fold_top_n"].iloc[0])
    if policy == "latest":
        return top_rows[top_rows["fold"].eq(latest_fold)].sort_values("score", ascending=False).head(latest_n)[["fold", "address", "score"]].copy()

    max_fold = max(active_folds)
    top_rows = top_rows.assign(
        rank_score=1.0 - ((top_rows["rank_in_fold"] - 1) / top_rows["fold_top_n"].clip(lower=1)),
        recency_weight=RECENCY_DECAY ** (max_fold - top_rows["fold"].astype(int)),
    )
    agg = top_rows.groupby("address", as_index=False).agg(
        fold=("fold", "max"),
        hits=("fold", "nunique"),
        mean_score=("score", "mean"),
        max_score=("score", "max"),
        mean_rank_score=("rank_score", "mean"),
    )
    recency = (
        top_rows.assign(weighted_rank_score=lambda x: x["rank_score"] * x["recency_weight"])
        .groupby("address", as_index=False)
        .agg(recency_score=("weighted_rank_score", "sum"), recency_weight=("recency_weight", "sum"))
    )
    recency["recency_score"] = recency["recency_score"] / recency["recency_weight"].clip(lower=1e-9)
    agg = agg.merge(recency[["address", "recency_score"]], on="address", how="left")
    if policy == "ensemble":
        agg["score"] = agg["mean_rank_score"] + 0.05 * np.log1p(agg["hits"])
        selected = agg.sort_values(["score", "max_score"], ascending=False).head(latest_n).copy()
    elif policy == "recency_decay":
        selected = agg.rename(columns={"recency_score": "score"}).sort_values(["score", "max_score"], ascending=False).head(latest_n).copy()
    else:
        raise ValueError(f"unsupported policy: {policy}")
    selected["fold"] = latest_fold
    return selected[["fold", "address", "score"]].copy()


def load_fold_windows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"fold", "future_start", "future_end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing fold-window columns: {sorted(missing)}")
    return df


def active_folds_for_day(fold_windows: pd.DataFrame | None, explicit_folds: list[int] | None, start_ts: int, end_ts: int) -> list[int]:
    if explicit_folds is not None:
        return explicit_folds
    if fold_windows is None:
        raise ValueError("predictions-like wallet source requires --active-folds or --fold-windows-csv")
    q = fold_windows[(fold_windows["future_end"] >= start_ts) & (fold_windows["future_start"] <= end_ts)]
    return sorted(pd.to_numeric(q["fold"], errors="coerce").dropna().astype(int).unique().tolist())


def load_wallets_by_date(args: argparse.Namespace, dates: list[str]) -> dict[str, pd.DataFrame]:
    src = Path(args.wallet_source)
    wallets = pd.read_csv(src, dtype={"address": str})
    if "address" not in wallets.columns:
        raise ValueError(f"{src} missing address column")
    wallets["address"] = wallets["address"].astype(str).str.lower()
    mode = args.wallet_source_mode
    if mode == "auto":
        mode = "predictions" if {"regime", "fold", "score"}.issubset(wallets.columns) and wallets["fold"].nunique() > 1 else "active"

    if mode == "active":
        required = {"fold", "address"}
        missing = required - set(wallets.columns)
        if missing:
            raise ValueError(f"active wallet source missing columns: {sorted(missing)}")
        if "score" not in wallets.columns:
            wallets["score"] = np.nan
        return {date: wallets[["fold", "address", "score"]].copy() for date in dates}

    required = {"fold", "address", "score"}
    missing = required - set(wallets.columns)
    if missing:
        raise ValueError(f"predictions wallet source missing columns: {sorted(missing)}")
    fold_windows = load_fold_windows(Path(args.fold_windows_csv)) if args.fold_windows_csv else None
    explicit = active_fold_list(args.active_folds)
    out = {}
    for date in dates:
        start_ts = date_epoch(date)
        end_ts = int((pd.Timestamp(date, tz="UTC") + pd.Timedelta(days=1)).timestamp())
        active = active_folds_for_day(fold_windows, explicit, start_ts, end_ts)
        out[date] = policy_wallets_from_predictions(wallets, active, args.policy, args.regime, args.threshold)
    return out


def classify_category(text: object) -> str:
    s = str(text or "").lower()
    rules = {
        "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "defi"],
        "politics_us": ["trump", "biden", "republican", "democrat", "congress", "senate", "election", "president"],
        "sports": ["nba", "nfl", "premier league", "ufc", "tennis", "basketball", "football", "soccer", "championship", "super bowl", "match", "game"],
        "finance": ["stock", "s&p", "nasdaq", "fed ", "interest rate", "inflation", "gdp", "price"],
        "ai_tech": [" ai ", "openai", "chatgpt", "apple", "google", "microsoft", "meta "],
    }
    for category, needles in rules.items():
        if any(needle in s for needle in needles):
            return category
    return "other"


def signal_side(direction: pd.Series, side: pd.Series) -> np.ndarray:
    direction = direction.astype(str).str.upper()
    side = side.astype(str).str.lower()
    token1 = ((direction == "BUY") & (side == "token1")) | ((direction == "SELL") & (side == "token2"))
    return np.where(token1, "token1", "token2")


def compute_delta(df: pd.DataFrame) -> pd.Series:
    direction = df["direction"].astype(str).str.upper()
    side = df["nonusdc_side"].astype(str).str.lower()
    price = pd.to_numeric(df["price"], errors="coerce")
    resolution = df["resolution"].astype(str).str.strip()
    answer1 = df["answer1"].astype(str).str.strip()
    answer2 = df["answer2"].astype(str).str.strip()
    correct = ((side == "token1") & resolution.eq(answer1)) | ((side == "token2") & resolution.eq(answer2))
    buy_delta = np.where(correct, 1.0 - price, -price)
    sell_delta = np.where(correct, price - 1.0, price)
    return pd.Series(np.where(direction == "BUY", buy_delta, sell_delta), index=df.index)


def expand_role(chunk: pd.DataFrame, role: str, wallets: pd.DataFrame) -> pd.DataFrame:
    address_col = role
    direction_col = f"{role}_direction"
    sub = pd.DataFrame(
        {
            "address": chunk[address_col].astype(str).str.lower(),
            "market_id": chunk["market_id"].astype(str),
            "price": pd.to_numeric(chunk["price"], errors="coerce"),
            "usd_amount": pd.to_numeric(chunk["usd_amount"], errors="coerce"),
            "direction": chunk[direction_col].astype(str).str.upper(),
            "nonusdc_side": chunk["nonusdc_side"].astype(str).str.lower(),
            "timestamp": chunk["_timestamp_epoch"],
        }
    )
    sub = sub[sub["address"].isin(set(wallets["address"]))]
    if sub.empty:
        return sub
    return sub.merge(wallets[["fold", "address"]], on="address", how="inner")


def write_outputs(args: argparse.Namespace, meta: pd.DataFrame, wallets_by_date: dict[str, pd.DataFrame], dates: list[str], start_ts: int, end_ts: int) -> list[dict[str, object]]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = suffix_for(args.policy)
    paths = {date: out_dir / f"daily_top_wallet_trades_{date.replace('-', '')}{suffix}.csv" for date in dates}
    active_paths = {date: out_dir / f"daily_active_top_wallets_{date.replace('-', '')}{suffix}.csv" for date in dates}
    summary = {
        date: {
            "date": date,
            "fold_policy": args.policy,
            "input_rows_in_processed_window": 0,
            "top_wallet_trades": 0,
            "unique_top_wallets": int(wallets_by_date[date]["address"].nunique()),
            "active_top_wallet_rows": int(len(wallets_by_date[date])),
            "active_folds": sorted(pd.to_numeric(wallets_by_date[date]["fold"], errors="coerce").dropna().astype(int).unique().tolist()),
            "unique_markets": 0,
            "missing_answer1_rows": 0,
            "missing_answer2_rows": 0,
            "missing_event_id_rows": 0,
            "missing_event_title_rows": 0,
            "metadata_complete_rows": 0,
            "resolved_trade_rows": 0,
            "missing_metadata_markets": 0,
        }
        for date in dates
    }
    market_sets = {date: set() for date in dates}
    market_trade_counts = {date: {} for date in dates}
    missing_market_stats = {date: {} for date in dates}
    written = {date: False for date in dates}

    for path in [*paths.values(), *active_paths.values()]:
        if path.exists():
            path.unlink()

    for date, wallets in wallets_by_date.items():
        wallets.to_csv(active_paths[date], index=False)

    for chunk in pd.read_csv(args.trades_csv, chunksize=args.chunksize, dtype=str):
        chunk["_timestamp_epoch"] = normalize_timestamp(chunk["timestamp"])
        chunk = chunk[chunk["_timestamp_epoch"].ge(start_ts) & chunk["_timestamp_epoch"].lt(end_ts)].copy()
        if chunk.empty:
            continue
        chunk["_date"] = pd.to_datetime(chunk["_timestamp_epoch"].astype("int64"), unit="s", utc=True).dt.strftime("%Y-%m-%d")
        for date in dates:
            day = chunk[chunk["_date"].eq(date)]
            summary[date]["input_rows_in_processed_window"] += int(len(day))
            if day.empty:
                continue
            wallets = wallets_by_date[date]
            roles = [expand_role(day, "maker", wallets), expand_role(day, "taker", wallets)]
            expanded = pd.concat([r for r in roles if not r.empty], ignore_index=True) if any(not r.empty for r in roles) else pd.DataFrame()
            if expanded.empty:
                continue
            expanded = expanded.merge(meta, on="market_id", how="left")
            expanded["signal_side"] = signal_side(expanded["direction"], expanded["nonusdc_side"])
            expanded["delta"] = compute_delta(expanded)
            expanded["dollar_delta"] = expanded["delta"] * expanded["usd_amount"]
            expanded["category"] = expanded["event_title"].map(classify_category)
            expanded = expanded[CACHE_COLUMNS]
            expanded.to_csv(paths[date], mode="a", header=not written[date], index=False)
            written[date] = True

            summary[date]["top_wallet_trades"] += int(len(expanded))
            market_sets[date].update(expanded["market_id"].dropna().astype(str).tolist())
            for mid, count in expanded["market_id"].dropna().astype(str).value_counts().items():
                market_trade_counts[date][mid] = market_trade_counts[date].get(mid, 0) + int(count)

            missing_by_col = {}
            for col in REQUIRED_METADATA_FOR_SIGNALS:
                missing_by_col[col] = expanded[col].isna() | expanded[col].astype(str).str.strip().eq("")
                summary[date][f"missing_{col}_rows"] += int(missing_by_col[col].sum())
            complete = pd.Series(True, index=expanded.index)
            for col in REQUIRED_METADATA_FOR_SIGNALS:
                complete &= expanded[col].notna() & expanded[col].astype(str).str.strip().ne("")
            summary[date]["metadata_complete_rows"] += int(complete.sum())
            summary[date]["resolved_trade_rows"] += int(expanded["resolution"].notna().sum() - expanded["resolution"].astype(str).str.strip().eq("").sum())

            missing_any = pd.Series(False, index=expanded.index)
            for mask in missing_by_col.values():
                missing_any |= mask
            if missing_any.any():
                for mid, group in expanded.loc[missing_any].groupby(expanded.loc[missing_any, "market_id"].astype(str), sort=False):
                    entry = missing_market_stats[date].setdefault(
                        mid,
                        {
                            "date": date,
                            "fold_policy": args.policy,
                            "market_id": mid,
                            "top_wallet_trade_rows": 0,
                            "missing_required_rows": 0,
                            **{f"missing_{col}_rows": 0 for col in REQUIRED_METADATA_FOR_SIGNALS},
                        },
                    )
                    entry["missing_required_rows"] += int(len(group))
                    for col in REQUIRED_METADATA_FOR_SIGNALS:
                        col_missing = group[col].isna() | group[col].astype(str).str.strip().eq("")
                        entry[f"missing_{col}_rows"] += int(col_missing.sum())

    for date in dates:
        if not written[date]:
            pd.DataFrame(columns=CACHE_COLUMNS).to_csv(paths[date], index=False)
        summary[date]["unique_markets"] = len(market_sets[date])
        for mid, entry in missing_market_stats[date].items():
            entry["top_wallet_trade_rows"] = market_trade_counts[date].get(mid, 0)
        summary[date]["missing_metadata_markets"] = len(missing_market_stats[date])
        summary[date]["resolved_trade_row_share"] = (
            summary[date]["resolved_trade_rows"] / summary[date]["top_wallet_trades"]
            if summary[date]["top_wallet_trades"]
            else 0.0
        )
        summary_path = out_dir / f"daily_summary_{date.replace('-', '')}{suffix}.json"
        summary_path.write_text(json.dumps(summary[date], indent=2, default=str))

    missing_rows = [
        entry
        for date in dates
        for entry in sorted(missing_market_stats[date].values(), key=lambda x: (-x["missing_required_rows"], x["market_id"]))
    ]
    missing_path = out_dir / f"late_window_adapter_missing_metadata_markets_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}_exclusive{suffix}.csv"
    pd.DataFrame(
        missing_rows,
        columns=[
            "date",
            "fold_policy",
            "market_id",
            "top_wallet_trade_rows",
            "missing_required_rows",
            *[f"missing_{col}_rows" for col in REQUIRED_METADATA_FOR_SIGNALS],
        ],
    ).to_csv(missing_path, index=False)
    return [summary[date] for date in dates]


def validate_schema(out_dir: Path, dates: list[str], policy: str, reference: Path | None) -> None:
    suffix = suffix_for(policy)
    expected = CACHE_COLUMNS
    if reference is not None:
        ref_cols = list(pd.read_csv(reference, nrows=0).columns)
        if ref_cols != expected:
            raise ValueError(f"internal CACHE_COLUMNS differ from reference {reference}: {ref_cols}")
    for date in dates:
        path = out_dir / f"daily_top_wallet_trades_{date.replace('-', '')}{suffix}.csv"
        cols = list(pd.read_csv(path, nrows=0).columns)
        if cols != expected:
            raise ValueError(f"{path} has columns {cols}, expected {expected}")


def main() -> None:
    args = parse_args()
    start_ts = date_epoch(args.start_date)
    end_ts = date_epoch(args.end_date)
    dates = pd.date_range(args.start_date, pd.Timestamp(args.end_date) - pd.Timedelta(days=1), freq="D").strftime("%Y-%m-%d").tolist()

    trades_csv = Path(args.trades_csv)
    market_ids, input_stats = collect_market_ids(trades_csv, start_ts, end_ts, args.chunksize)
    meta = load_metadata([Path(p) for p in args.metadata_csv], market_ids, args.chunksize)
    resolutions = load_resolutions(Path(args.resolutions_csv)) if args.resolutions_csv else None
    meta = merge_resolutions(meta, resolutions)
    wallets_by_date = load_wallets_by_date(args, dates)
    rows = write_outputs(args, meta, wallets_by_date, dates, start_ts, end_ts)
    validate_schema(Path(args.out_dir), dates, args.policy, Path(args.schema_reference_csv) if args.schema_reference_csv else None)

    coverage_path = Path(args.out_dir) / f"late_window_adapter_coverage_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}_exclusive.csv"
    missing_metadata_path = Path(args.out_dir) / (
        f"late_window_adapter_missing_metadata_markets_"
        f"{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}_exclusive{suffix_for(args.policy)}.csv"
    )
    coverage = pd.DataFrame(rows)
    coverage["input_rows_in_processed_window_total"] = input_stats["input_rows_in_window"]
    coverage.to_csv(coverage_path, index=False)

    missing_cols = [f"missing_{col}_rows" for col in REQUIRED_METADATA_FOR_SIGNALS]
    bad = coverage[coverage[missing_cols].sum(axis=1).gt(0)]
    if not bad.empty and not args.allow_missing_metadata:
        raise SystemExit(
            "metadata coverage gate failed for emitted top-wallet rows; "
            f"see {coverage_path} and {missing_metadata_path}. "
            "Re-run with --allow-missing-metadata only for diagnostics."
        )
    print(coverage.to_string(index=False))
    print(f"Wrote {coverage_path}")


if __name__ == "__main__":
    main()
