"""
Utilities for integrating external Polymarket datasets.

These helpers keep source-specific schemas out of the core preprocessing
pipeline. The canonical transaction contract remains:
transaction_id, address, market_id, side, outcome, amount, price, timestamp.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


CANONICAL_TRANSACTION_COLUMNS = [
    "transaction_id",
    "address",
    "market_id",
    "side",
    "outcome",
    "amount",
    "price",
    "timestamp",
]

MARKET_RESOLUTION_COLUMNS = [
    "condition_id",
    "market_id",
    "winner_token_id",
    "winner_outcome",
    "resolution_source",
    "resolution_confidence",
    "is_phase26_primary",
    "is_secondary_fallback",
    "terminal_price_max",
    "terminal_price_second",
    "terminal_price_gap",
    "token_ids_json",
    "question",
    "resolution",
    "closed",
    "winning_outcome",
    "winning_outcome_index",
    "winning_outcome_price",
    "outcomes_json",
    "outcome_prices_json",
    "slug",
    "resolution_method",
    "resolved_at",
]


PRIMARY_RESOLUTION_SOURCE = "clob_winner_field"
SECONDARY_RESOLUTION_SOURCE = "clob_terminal_price_argmax_secondary"


def _as_utc_naive(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)


def _numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _parse_tag_slugs(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    try:
        tags = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(tags, list):
        return ""
    slugs = []
    for tag in tags:
        if isinstance(tag, dict):
            slug = tag.get("slug") or tag.get("label")
            if slug:
                slugs.append(str(slug))
    return "|".join(slugs)


def _topic_slug(tag_slugs: object) -> str | None:
    generic = {"", "all", "featured", "new", "markets"}
    for slug in str(tag_slugs or "").split("|"):
        clean = slug.strip()
        if clean.lower() not in generic:
            return clean
    return None


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def build_failed_clob_resolution_record(market_id: str, method: str = "fetch_failed") -> dict:
    """Build a schema-compatible CLOB resolution failure record."""
    return {
        "condition_id": market_id,
        "market_id": market_id,
        "winner_token_id": None,
        "winner_outcome": "",
        "resolution_source": method,
        "resolution_confidence": "unresolved",
        "is_phase26_primary": False,
        "is_secondary_fallback": False,
        "terminal_price_max": None,
        "terminal_price_second": None,
        "terminal_price_gap": None,
        "question": "",
        "resolution": "__UNFETCHED__",
        "closed": False,
        "winning_outcome": "",
        "winning_outcome_index": None,
        "winning_outcome_price": 0.0,
        "outcomes_json": "",
        "outcome_prices_json": "",
        "slug": "",
        "resolution_method": method,
        "resolved_at": "",
    }


def _token_id(token: dict) -> str | None:
    for key in ("token_id", "tokenId", "id"):
        value = token.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _terminal_price_stats(prices: list[float]) -> tuple[float | None, float | None, float | None]:
    clean = sorted([p for p in prices if pd.notna(p)], reverse=True)
    if not clean:
        return None, None, None
    max_price = float(clean[0])
    second_price = float(clean[1]) if len(clean) > 1 else 0.0
    return max_price, second_price, float(max_price - second_price)


def _with_resolution_flags(record: dict, source: str, confidence: str) -> dict:
    record["resolution_source"] = source
    record["resolution_method"] = source
    record["resolution_confidence"] = confidence
    record["is_phase26_primary"] = source == PRIMARY_RESOLUTION_SOURCE
    record["is_secondary_fallback"] = source == SECONDARY_RESOLUTION_SOURCE
    return record


def parse_clob_market_resolution(market_id: str, data: dict) -> dict:
    """Parse a CLOB market payload into the project resolution contract.

    This intentionally trusts only the explicit `tokens[].winner` field.
    It does not infer winners from terminal prices or Gamma metadata.
    """
    tokens = data.get("tokens") or []
    outcomes = [str(t.get("outcome", "")) for t in tokens if isinstance(t, dict)]
    token_ids = [_token_id(t) for t in tokens if isinstance(t, dict)]
    prices = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        try:
            prices.append(float(token.get("price", 0.0)))
        except (TypeError, ValueError):
            prices.append(0.0)

    closed = bool(data.get("closed", False))
    winner_idx = next(
        (idx for idx, token in enumerate(tokens) if isinstance(token, dict) and bool(token.get("winner"))),
        None,
    )
    max_price, second_price, price_gap = _terminal_price_stats(prices)
    if winner_idx is not None and closed:
        winner_token = tokens[winner_idx]
        winning_outcome = str(winner_token.get("outcome", ""))
        try:
            winning_price = float(winner_token.get("price", 1.0))
        except (TypeError, ValueError):
            winning_price = 1.0
        return _with_resolution_flags({
            "condition_id": market_id,
            "market_id": market_id,
            "winner_token_id": _token_id(winner_token),
            "winner_outcome": winning_outcome,
            "terminal_price_max": max_price,
            "terminal_price_second": second_price,
            "terminal_price_gap": price_gap,
            "question": data.get("question", ""),
            "resolution": winning_outcome,
            "closed": True,
            "winning_outcome": winning_outcome,
            "winning_outcome_index": winner_idx,
            "winning_outcome_price": winning_price,
            "outcomes_json": json.dumps(outcomes, ensure_ascii=False),
            "outcome_prices_json": json.dumps(prices),
            "slug": data.get("market_slug", data.get("slug", "")),
            "resolved_at": data.get("end_date_iso", data.get("closed_time", "")),
            "token_ids_json": json.dumps(token_ids),
        }, PRIMARY_RESOLUTION_SOURCE, "primary")

    return _with_resolution_flags({
        "condition_id": market_id,
        "market_id": market_id,
        "winner_token_id": None,
        "winner_outcome": "",
        "terminal_price_max": max_price,
        "terminal_price_second": second_price,
        "terminal_price_gap": price_gap,
        "question": data.get("question", ""),
        "resolution": "__OPEN__",
        "closed": closed,
        "winning_outcome": "",
        "winning_outcome_index": None,
        "winning_outcome_price": 0.0,
        "outcomes_json": json.dumps(outcomes, ensure_ascii=False),
        "outcome_prices_json": json.dumps(prices),
        "slug": data.get("market_slug", data.get("slug", "")),
        "resolved_at": "",
        "token_ids_json": json.dumps(token_ids),
    }, "clob_no_winner", "unresolved")


def apply_secondary_terminal_price_fallback(record: dict) -> dict:
    """Recover a resolution from decisive CLOB terminal prices when no winner exists."""
    source = record.get("resolution_source", record.get("resolution_method", ""))
    if source == PRIMARY_RESOLUTION_SOURCE:
        return _with_resolution_flags(dict(record), PRIMARY_RESOLUTION_SOURCE, "primary")

    out = dict(record)
    try:
        prices = json.loads(out.get("outcome_prices_json", "[]"))
        outcomes = json.loads(out.get("outcomes_json", "[]"))
    except (TypeError, json.JSONDecodeError):
        out.update({"winner_token_id": None, "winner_outcome": ""})
        return _with_resolution_flags(out, "unresolved_parse_error", "unresolved")

    if not isinstance(prices, list) or not isinstance(outcomes, list) or len(prices) == 0 or len(prices) != len(outcomes):
        out.update({"winner_token_id": None, "winner_outcome": ""})
        return _with_resolution_flags(out, "unresolved_missing_tokens", "unresolved")

    token_ids_raw = out.get("token_ids_json", "[]")
    try:
        token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
    except (TypeError, json.JSONDecodeError):
        token_ids = []
    if not isinstance(token_ids, list) or len(token_ids) != len(prices):
        token_ids = [None] * len(prices)

    try:
        price_values = [float(price) for price in prices]
    except (TypeError, ValueError):
        out.update({"winner_token_id": None, "winner_outcome": ""})
        return _with_resolution_flags(out, "unresolved_parse_error", "unresolved")

    max_price, second_price, price_gap = _terminal_price_stats(price_values)
    out["terminal_price_max"] = max_price
    out["terminal_price_second"] = second_price
    out["terminal_price_gap"] = price_gap

    if max_price is None or second_price is None:
        out.update({"winner_token_id": None, "winner_outcome": ""})
        return _with_resolution_flags(out, "unresolved_non_decisive", "unresolved")
    if price_values.count(max_price) != 1 or max_price < 0.99 or price_gap < 0.98:
        out.update({"winner_token_id": None, "winner_outcome": ""})
        return _with_resolution_flags(out, "unresolved_non_decisive", "unresolved")

    winner_idx = price_values.index(max_price)
    winner_outcome = str(outcomes[winner_idx])
    out.update(
        {
            "winner_token_id": token_ids[winner_idx],
            "winner_outcome": winner_outcome,
            "resolution": winner_outcome,
            "winning_outcome": winner_outcome,
            "winning_outcome_index": winner_idx,
            "winning_outcome_price": max_price,
            "closed": True,
        }
    )
    return _with_resolution_flags(out, SECONDARY_RESOLUTION_SOURCE, "secondary")


def build_resolution_coverage_summary(resolutions: pd.DataFrame, primary_gate: float = 0.90) -> dict:
    """Summarize primary and secondary resolution coverage without mixing them."""
    if resolutions.empty:
        total = 0
    elif "market_id" in resolutions.columns:
        total = int(resolutions["market_id"].dropna().nunique())
    else:
        total = int(len(resolutions))
    source_col = "resolution_source" if "resolution_source" in resolutions.columns else "resolution_method"
    sources = resolutions[source_col].fillna("").astype(str) if source_col in resolutions.columns else pd.Series([], dtype=str)
    primary = int((sources == PRIMARY_RESOLUTION_SOURCE).sum())
    secondary = int((sources == SECONDARY_RESOLUTION_SOURCE).sum())
    unresolved_non_decisive = int((sources == "unresolved_non_decisive").sum())
    unresolved_parse_error = int((sources == "unresolved_parse_error").sum())
    unresolved_missing_tokens = int((sources == "unresolved_missing_tokens").sum())
    clob_no_winner = int((sources == "clob_no_winner").sum())
    remaining = total - primary - secondary
    original_no_winner = clob_no_winner if clob_no_winner else secondary + remaining
    return {
        "sample_markets": total,
        "clob_winner_field": primary,
        "clob_no_winner": original_no_winner,
        "secondary_terminal_price_recovered": secondary,
        "secondary_terminal_price_non_decisive": unresolved_non_decisive,
        "unresolved_parse_error": unresolved_parse_error,
        "unresolved_missing_tokens": unresolved_missing_tokens,
        "primary_clob_winner_field_coverage": primary / total if total else 0.0,
        "secondary_recoverable_coverage": secondary / total if total else 0.0,
        "combined_usable_resolution_coverage": (primary + secondary) / total if total else 0.0,
        "remaining_unresolved_coverage": remaining / total if total else 0.0,
        "phase26_primary_gate_passed": (primary / total if total else 0.0) >= primary_gate,
    }


def build_event_context_features(events_csv: str | Path) -> pd.DataFrame:
    """Build compact event-level metadata features from Polymarket events CSV."""
    df = pd.read_csv(events_csv, low_memory=False)
    out = pd.DataFrame()
    out["event_id"] = df["id"].astype("string")
    out["event_slug"] = df.get("slug", pd.Series(index=df.index, dtype="string")).astype("string")
    out["event_title"] = df.get("title", pd.Series(index=df.index, dtype="string")).astype("string")
    out["series_slug"] = df.get("seriesSlug", pd.Series(index=df.index, dtype="string")).astype("string")
    out["category"] = df.get("category", pd.Series(index=df.index, dtype="string")).astype("string")
    out["tag_slugs"] = df.get("tags", pd.Series(index=df.index, dtype="string")).apply(_parse_tag_slugs)
    out["primary_tag_slug"] = out["tag_slugs"].astype(str).str.split("|").str[0].replace("", pd.NA)
    out["topic_slug"] = out["tag_slugs"].apply(_topic_slug)

    start = _as_utc_naive(df.get("startDate", pd.Series(index=df.index)))
    end = _as_utc_naive(df.get("endDate", pd.Series(index=df.index)))
    closed_time = _as_utc_naive(df.get("closedTime", pd.Series(index=df.index)))
    created = _as_utc_naive(df.get("createdAt", pd.Series(index=df.index)))

    out["created_at"] = created
    out["start_date"] = start
    out["end_date"] = end
    out["closed_time"] = closed_time
    out["duration_days"] = ((end - start).dt.total_seconds() / 86400.0).clip(lower=0)
    out["creation_to_start_days"] = ((start - created).dt.total_seconds() / 86400.0)
    out["volume"] = _numeric(df.get("volume", pd.Series(index=df.index)))
    out["volume_24h"] = _numeric(df.get("volume24hr", pd.Series(index=df.index)))
    out["volume_1w"] = _numeric(df.get("volume1wk", pd.Series(index=df.index)))
    out["volume_1mo"] = _numeric(df.get("volume1mo", pd.Series(index=df.index)))
    out["liquidity"] = _numeric(df.get("liquidity", pd.Series(index=df.index)))
    out["liquidity_clob"] = _numeric(df.get("liquidityClob", pd.Series(index=df.index)))
    out["comment_count"] = _numeric(df.get("commentCount", pd.Series(index=df.index)))
    out["open_interest"] = _numeric(df.get("openInterest", pd.Series(index=df.index)))
    out["market_count"] = _numeric(df.get("market_count", pd.Series(index=df.index)))
    out["closed"] = _bool_series(df.get("closed", pd.Series(False, index=df.index)))
    out["active"] = _bool_series(df.get("active", pd.Series(False, index=df.index)))
    out["enable_order_book"] = _bool_series(
        df.get("enableOrderBook", pd.Series(False, index=df.index))
    )
    out["enable_neg_risk"] = _bool_series(
        df.get("enableNegRisk", pd.Series(False, index=df.index))
    )
    out["competitive"] = _numeric(df.get("competitive", pd.Series(index=df.index)))
    return out


def summarize_event_metadata(events: pd.DataFrame) -> dict:
    """Return a small JSON-serializable profile for event metadata."""
    date_col = "start_date" if "start_date" in events.columns else None
    return {
        "rows": int(len(events)),
        "unique_event_ids": int(events["event_id"].nunique(dropna=True)) if "event_id" in events else 0,
        "closed_events": int(events["closed"].sum()) if "closed" in events else 0,
        "active_events": int(events["active"].sum()) if "active" in events else 0,
        "series_count": int(events["series_slug"].nunique(dropna=True)) if "series_slug" in events else 0,
        "category_count": int(events["category"].nunique(dropna=True)) if "category" in events else 0,
        "start_min": str(events[date_col].min()) if date_col else None,
        "start_max": str(events[date_col].max()) if date_col else None,
        "total_volume": float(events["volume"].sum()) if "volume" in events else 0.0,
    }


def convert_sii_users_to_transactions(
    users: pd.DataFrame,
    *,
    prefer_condition_id: bool = True,
) -> pd.DataFrame:
    """Convert SII `users.parquet` rows to this project's transaction schema."""
    required = {"transaction_hash", "token_amount", "price"}
    missing = sorted(required - set(users.columns))
    if missing:
        raise ValueError(f"SII users data missing required columns: {missing}")
    address_col = "user" if "user" in users.columns else "address" if "address" in users.columns else None
    if address_col is None:
        raise ValueError("SII users data needs either user or address")

    df = users.copy()
    if prefer_condition_id and "condition_id" in df.columns:
        market_id = df["condition_id"]
    elif "market_id" in df.columns:
        market_id = df["market_id"]
    else:
        raise ValueError("SII users data needs either condition_id or market_id")

    log_index = df["log_index"].astype(str) if "log_index" in df.columns else "0"
    role = df["role"].astype(str) if "role" in df.columns else "user"
    token_amount = _numeric(df["token_amount"])

    timestamp_source = df["timestamp"] if "timestamp" in df.columns else df.get("datetime")
    if timestamp_source is None:
        raise ValueError("SII users data needs timestamp or datetime")
    if pd.api.types.is_numeric_dtype(timestamp_source):
        timestamp = pd.to_datetime(timestamp_source, unit="s", errors="coerce", utc=True).dt.tz_convert(None)
    else:
        timestamp = _as_utc_naive(timestamp_source)
    if "direction" in df.columns:
        side = df["direction"].astype(str).str.upper().where(
            df["direction"].astype(str).str.upper().isin(["BUY", "SELL"]),
            token_amount.ge(0).map({True: "BUY", False: "SELL"}),
        )
    else:
        side = token_amount.ge(0).map({True: "BUY", False: "SELL"})

    out = pd.DataFrame(
        {
            "transaction_id": (
                df["transaction_hash"].astype(str)
                + ":"
                + log_index.astype(str)
                + ":"
                + role.astype(str)
                + ":"
                + df[address_col].astype(str)
            ),
            "address": df[address_col].astype(str),
            "market_id": market_id.astype(str),
            "side": side,
            "outcome": df["answer"].astype(str) if "answer" in df.columns else None,
            "amount": token_amount.abs(),
            "price": _numeric(df["price"]),
            "timestamp": timestamp,
        }
    )
    return out[CANONICAL_TRANSACTION_COLUMNS].dropna(subset=["transaction_id", "address", "market_id", "timestamp"])


def convert_sii_trades_to_transactions(
    trades: pd.DataFrame,
    *,
    prefer_condition_id: bool = True,
) -> pd.DataFrame:
    """Convert SII `trades.parquet` rows by splitting maker and taker wallets."""
    required = {"transaction_hash", "maker", "taker", "price"}
    missing = sorted(required - set(trades.columns))
    if missing:
        raise ValueError(f"SII trades data missing required columns: {missing}")
    rows = []
    for role, address_col in [("maker", "maker"), ("taker", "taker")]:
        part = trades.copy()
        part["user"] = part[address_col]
        part["role"] = role
        if "token_amount" not in part.columns:
            part["token_amount"] = part.get("amount", part.get("usd_amount", 0))
        rows.append(convert_sii_users_to_transactions(part, prefer_condition_id=prefer_condition_id))
    return pd.concat(rows, ignore_index=True)


def compare_event_market_ids(events: pd.DataFrame, markets: pd.DataFrame) -> dict:
    """Profile likely join keys between local events CSV and SII markets metadata."""
    result = {"events_rows": int(len(events)), "markets_rows": int(len(markets))}
    comparisons: Iterable[tuple[str, str]] = [
        ("event_id", "event_id"),
        ("event_id", "id"),
        ("event_slug", "event_slug"),
        ("event_slug", "slug"),
        ("event_slug", "event_slug"),
    ]
    for left, right in comparisons:
        if left in events.columns and right in markets.columns:
            lvals = set(events[left].dropna().astype(str))
            rvals = set(markets[right].dropna().astype(str))
            key = f"{left}__to__{right}"
            overlap = len(lvals & rvals)
            result[key] = {
                "left_unique": len(lvals),
                "right_unique": len(rvals),
                "overlap": overlap,
                "left_overlap_rate": overlap / len(lvals) if lvals else 0.0,
                "right_overlap_rate": overlap / len(rvals) if rvals else 0.0,
            }
    return result


def build_sii_market_event_map(
    transactions: pd.DataFrame,
    markets: pd.DataFrame,
    event_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map project market ids to SII market metadata and local event context."""
    if "market_id" not in transactions.columns:
        raise ValueError("transactions must include market_id")
    if "condition_id" not in markets.columns:
        raise ValueError("markets metadata must include condition_id")

    sample_markets = pd.DataFrame(
        {"market_id": transactions["market_id"].dropna().astype(str).unique()}
    )
    market_cols = [
        col
        for col in [
            "id",
            "condition_id",
            "question",
            "slug",
            "event_id",
            "event_slug",
            "event_title",
            "created_at",
            "end_date",
            "updated_at",
            "closed",
            "active",
            "archived",
            "outcome_prices",
            "volume",
            "neg_risk",
        ]
        if col in markets.columns
    ]
    market_meta = markets[market_cols].copy()
    market_meta["condition_id"] = market_meta["condition_id"].astype(str)
    out = sample_markets.merge(
        market_meta,
        left_on="market_id",
        right_on="condition_id",
        how="left",
        suffixes=("", "_sii"),
    )
    out = out.rename(
        columns={
            "id": "sii_market_id",
            "slug": "market_slug",
            "volume": "market_volume",
            "created_at": "market_created_at",
            "end_date": "market_end_date",
            "updated_at": "market_updated_at",
            "closed": "market_closed",
            "active": "market_active",
            "archived": "market_archived",
            "neg_risk": "market_neg_risk",
        }
    )
    if event_features is not None and not event_features.empty:
        events = event_features.copy()
        events["event_id"] = events["event_id"].astype(str)
        out["event_id"] = out["event_id"].astype("string")
        out = out.merge(
            events,
            on="event_id",
            how="left",
            suffixes=("_sii", ""),
        )
        local_signal_cols = [col for col in ["event_title", "category", "tag_slugs"] if col in out.columns]
        out["has_local_event_metadata"] = out[local_signal_cols].notna().any(axis=1) if local_signal_cols else False
        for col in ["event_slug", "event_title"]:
            sii_col = f"{col}_sii"
            if sii_col in out.columns and col in out.columns:
                out[col] = out[col].fillna(out[sii_col])
                out = out.drop(columns=[sii_col])
    else:
        out["has_local_event_metadata"] = False
    return out


def summarize_market_event_map(market_event_map: pd.DataFrame) -> dict:
    """Summarize market-to-event join coverage."""
    total = int(len(market_event_map))
    matched_market = (
        int(market_event_map["sii_market_id"].notna().sum())
        if "sii_market_id" in market_event_map
        else 0
    )
    matched_event = (
        int(market_event_map["has_local_event_metadata"].fillna(False).sum())
        if "has_local_event_metadata" in market_event_map
        else 0
    )
    return {
        "sample_markets": total,
        "sii_market_metadata_matched": matched_market,
        "sii_market_metadata_coverage": matched_market / total if total else 0.0,
        "local_event_metadata_matched": matched_event,
        "local_event_metadata_coverage": matched_event / total if total else 0.0,
        "unique_events": int(market_event_map["event_id"].nunique(dropna=True))
        if "event_id" in market_event_map
        else 0,
        "unique_event_slugs": int(market_event_map["event_slug"].nunique(dropna=True))
        if "event_slug" in market_event_map
        else 0,
        "unique_categories": int(market_event_map["category"].nunique(dropna=True))
        if "category" in market_event_map
        else 0,
    }


def build_wallet_event_features(
    transactions: pd.DataFrame,
    market_event_map: pd.DataFrame,
) -> pd.DataFrame:
    """Build pre-resolution wallet features from event/category metadata."""
    required = {"address", "market_id", "amount", "price"}
    missing = sorted(required - set(transactions.columns))
    if missing:
        raise ValueError(f"transactions missing required columns: {missing}")

    map_cols = [
        col
        for col in [
            "market_id",
            "event_id",
            "event_slug",
            "event_title",
            "category",
            "series_slug",
            "tag_slugs",
            "primary_tag_slug",
            "topic_slug",
            "market_count",
            "volume",
            "liquidity",
            "liquidity_clob",
            "comment_count",
            "open_interest",
            "duration_days",
            "creation_to_start_days",
            "market_volume",
            "market_neg_risk",
            "enable_order_book",
            "enable_neg_risk",
            "competitive",
        ]
        if col in market_event_map.columns
    ]
    enriched = transactions.merge(
        market_event_map[map_cols].drop_duplicates("market_id"),
        on="market_id",
        how="left",
    )
    enriched["trade_notional"] = _numeric(enriched["amount"]) * _numeric(enriched["price"])
    enriched["event_id"] = enriched["event_id"].fillna("__missing_event__").astype(str)
    enriched["category"] = enriched["category"].fillna("__missing_category__").astype(str)
    if "primary_tag_slug" in enriched.columns:
        enriched["primary_tag_slug"] = (
            enriched["primary_tag_slug"].fillna("__missing_tag__").astype(str)
        )
    if "topic_slug" in enriched.columns:
        enriched["topic_slug"] = enriched["topic_slug"].fillna("__missing_topic__").astype(str)

    def _hhi(values: pd.Series) -> float:
        weights = _numeric(values)
        total = float(weights.sum())
        if total <= 0:
            return 0.0
        shares = weights / total
        return float((shares * shares).sum())

    def _top_share(values: pd.Series) -> float:
        weights = _numeric(values)
        total = float(weights.sum())
        if total <= 0:
            return 0.0
        return float(weights.max() / total)

    rows = []
    for address, group in enriched.groupby("address"):
        event_notional = group.groupby("event_id")["trade_notional"].sum()
        series_notional = (
            group.groupby("series_slug")["trade_notional"].sum()
            if "series_slug" in group
            else pd.Series(dtype=float)
        )
        category_notional = group.groupby("category")["trade_notional"].sum()
        primary_tag_notional = (
            group.groupby("primary_tag_slug")["trade_notional"].sum()
            if "primary_tag_slug" in group
            else pd.Series(dtype=float)
        )
        topic_notional = (
            group.groupby("topic_slug")["trade_notional"].sum()
            if "topic_slug" in group
            else pd.Series(dtype=float)
        )
        event_market_counts = group.groupby("event_id")["market_id"].nunique()
        trade_notional = _numeric(group["trade_notional"])
        total_notional = float(trade_notional.sum())
        neg_risk = _bool_series(group.get("market_neg_risk", pd.Series(False, index=group.index)))
        order_book = _bool_series(group.get("enable_order_book", pd.Series(False, index=group.index)))
        neg_risk_notional = float(trade_notional[neg_risk].sum())
        order_book_notional = float(trade_notional[order_book].sum())
        rows.append(
            {
                "address": address,
                "event_trade_count": int(len(group)),
                "unique_events": int(group["event_id"].nunique()),
                "unique_event_slugs": int(group["event_slug"].nunique(dropna=True))
                if "event_slug" in group
                else 0,
                "unique_series": int(group["series_slug"].nunique(dropna=True))
                if "series_slug" in group
                else 0,
                "unique_event_categories": int(group["category"].nunique()),
                "unique_primary_tags": int(group["primary_tag_slug"].nunique(dropna=True))
                if "primary_tag_slug" in group
                else 0,
                "unique_topics": int(group["topic_slug"].nunique(dropna=True))
                if "topic_slug" in group
                else 0,
                "event_diversification": float(group["event_id"].nunique() / len(group))
                if len(group)
                else 0.0,
                "series_diversification": float(group["series_slug"].nunique(dropna=True) / len(group))
                if "series_slug" in group and len(group)
                else 0.0,
                "topic_diversification": float(group["topic_slug"].nunique(dropna=True) / len(group))
                if "topic_slug" in group and len(group)
                else 0.0,
                "event_notional_hhi": _hhi(event_notional),
                "top_event_notional_share": _top_share(event_notional),
                "series_notional_hhi": _hhi(series_notional),
                "top_series_notional_share": _top_share(series_notional),
                "category_notional_hhi": _hhi(category_notional),
                "top_category_notional_share": _top_share(category_notional),
                "primary_tag_notional_hhi": _hhi(primary_tag_notional),
                "top_primary_tag_notional_share": _top_share(primary_tag_notional),
                "topic_notional_hhi": _hhi(topic_notional),
                "top_topic_notional_share": _top_share(topic_notional),
                "same_event_multi_market_share": float((event_market_counts > 1).mean())
                if len(event_market_counts)
                else 0.0,
                "mean_markets_per_event_traded": float(event_market_counts.mean())
                if len(event_market_counts)
                else 0.0,
                "avg_event_market_count": float(_numeric(group.get("market_count", pd.Series())).mean())
                if "market_count" in group
                else 0.0,
                "avg_event_volume": float(_numeric(group.get("volume", pd.Series())).mean())
                if "volume" in group
                else 0.0,
                "avg_event_liquidity": float(_numeric(group.get("liquidity", pd.Series())).mean())
                if "liquidity" in group
                else 0.0,
                "avg_event_duration_days": float(_numeric(group.get("duration_days", pd.Series())).mean())
                if "duration_days" in group
                else 0.0,
                "avg_creation_to_start_days": float(
                    _numeric(group.get("creation_to_start_days", pd.Series())).mean()
                )
                if "creation_to_start_days" in group
                else 0.0,
                "avg_event_comment_count": float(_numeric(group.get("comment_count", pd.Series())).mean())
                if "comment_count" in group
                else 0.0,
                "avg_event_open_interest": float(_numeric(group.get("open_interest", pd.Series())).mean())
                if "open_interest" in group
                else 0.0,
                "neg_risk_trade_share": float(neg_risk.mean()) if len(neg_risk) else 0.0,
                "neg_risk_notional_share": neg_risk_notional / total_notional
                if total_notional > 0
                else 0.0,
                "order_book_trade_share": float(order_book.mean()) if len(order_book) else 0.0,
                "order_book_notional_share": order_book_notional / total_notional
                if total_notional > 0
                else 0.0,
                "avg_competitive_score": float(_numeric(group.get("competitive", pd.Series())).mean())
                if "competitive" in group
                else 0.0,
            }
        )
    return pd.DataFrame(rows).fillna(0)
