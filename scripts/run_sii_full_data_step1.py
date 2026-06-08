#!/usr/bin/env python
"""Step 1 full-data validation on SII Polymarket users.parquet.

This runner avoids loading the raw 100GB-scale dataset into pandas. DuckDB scans
the parquet file and materializes a compact wallet-level panel, then sklearn /
LightGBM train the three focused feature regimes requested for Step 1.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover
    hf_hub_download = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LIVE_COLS = [
    "early_entry_score",
    "contrarian_score",
    "information_ratio",
    "cross_market_diversification",
    "avg_holding_period",
    "trading_frequency",
    "capital_flow_centrality",
]

TOPIC_COLS = [
    "unique_topics",
    "topic_diversification",
    "topic_notional_hhi",
    "top_topic_notional_share",
]

STATIC_EVENT_COLS = [
    "avg_event_market_count",
    "avg_event_duration_days",
    "avg_creation_to_start_days",
    "neg_risk_trade_share",
    "order_book_trade_share",
]

FEATURE_REGIMES = {
    "live": LIVE_COLS,
    "live_plus_topic_diversification": LIVE_COLS + TOPIC_COLS,
    "live_plus_static_event_shape": LIVE_COLS + STATIC_EVENT_COLS,
}


@dataclass
class RunConfig:
    min_trades: int
    min_volume: float
    observation_days: int
    horizon_days: int
    top_label_percentile: float
    rolling_folds: int
    random_seeds: int
    top_select_percentile: float


def _download_missing(path: Path, filename: str) -> Path:
    if path.exists():
        return path
    if hf_hub_download is None:
        raise SystemExit("huggingface_hub is not installed and input file is missing")
    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id="SII-WANGZJ/Polymarket_data",
        repo_type="dataset",
        filename=filename,
        local_dir=str(path.parent),
        local_dir_use_symlinks=False,
    )
    downloaded_path = Path(downloaded)
    if downloaded_path != path and not path.exists():
        path.write_bytes(downloaded_path.read_bytes())
    return path


def _parse_prices(raw: object) -> list[float]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[float] = []
    for value in raw:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            return []
    return out


def build_market_context(markets_path: Path, events_csv: Path, out_dir: Path) -> tuple[Path, Path, dict]:
    """Build full-market terminal resolutions and event metadata context."""
    out_dir.mkdir(parents=True, exist_ok=True)
    resolution_path = out_dir / "sii_full_market_resolutions.parquet"
    context_path = out_dir / "sii_full_market_event_context.parquet"

    markets = pd.read_parquet(markets_path)
    prices = markets["outcome_prices"].apply(_parse_prices)
    answer_cols = [col for col in ["answer1", "answer2"] if col in markets.columns]

    winners = []
    sources = []
    max_prices = []
    gaps = []
    for idx, vals in prices.items():
        if len(vals) < 2 or not answer_cols:
            winners.append(None)
            sources.append("unresolved_missing_prices")
            max_prices.append(np.nan)
            gaps.append(np.nan)
            continue
        order = sorted(vals, reverse=True)
        max_price = float(order[0])
        second = float(order[1]) if len(order) > 1 else 0.0
        gap = max_price - second
        max_prices.append(max_price)
        gaps.append(gap)
        if vals.count(max_price) == 1 and max_price >= 0.99 and gap >= 0.98:
            winner_idx = vals.index(max_price)
            col = f"answer{winner_idx + 1}"
            winners.append(markets.at[idx, col] if col in markets.columns else None)
            sources.append("terminal_price_argmax_full")
        else:
            winners.append(None)
            sources.append("unresolved_non_decisive")

    resolutions = pd.DataFrame(
        {
            "market_id": markets["condition_id"].astype(str),
            "resolution": pd.Series(winners, dtype="object").astype("string"),
            "resolution_source": sources,
            "terminal_price_max": max_prices,
            "terminal_price_gap": gaps,
        }
    )
    resolutions = resolutions.dropna(subset=["resolution"])
    resolutions.to_parquet(resolution_path, index=False)

    context_cols = [
        "condition_id",
        "token1",
        "token2",
        "answer1",
        "answer2",
        "event_id",
        "event_slug",
        "event_title",
        "question",
        "volume",
        "neg_risk",
    ]
    context_cols = [col for col in context_cols if col in markets.columns]
    context = markets[context_cols].copy()
    context = context.rename(
        columns={
            "condition_id": "market_id",
            "volume": "market_volume",
            "neg_risk": "market_neg_risk",
        }
    )
    context["market_id"] = context["market_id"].astype(str)
    context["event_id"] = context["event_id"].astype("string")

    if events_csv.exists():
        events = pd.read_csv(events_csv, low_memory=False)
        event_context = pd.DataFrame()
        event_context["event_id"] = events["id"].astype("string")
        event_context["series_slug"] = events.get("seriesSlug", pd.Series(index=events.index, dtype="string")).astype("string")
        event_context["tag_slugs"] = events.get("tags", pd.Series(index=events.index, dtype="string")).apply(_parse_tag_slugs)
        event_context["topic_slug"] = event_context["tag_slugs"].apply(_topic_slug).fillna("__missing_topic__")
        event_context["created_at"] = _as_dt(events.get("createdAt", pd.Series(index=events.index)))
        event_context["start_date"] = _as_dt(events.get("startDate", pd.Series(index=events.index)))
        event_context["end_date"] = _as_dt(events.get("endDate", pd.Series(index=events.index)))
        event_context["duration_days"] = (
            (event_context["end_date"] - event_context["start_date"]).dt.total_seconds() / 86400.0
        ).clip(lower=0)
        event_context["creation_to_start_days"] = (
            (event_context["start_date"] - event_context["created_at"]).dt.total_seconds() / 86400.0
        )
        event_context["market_count"] = pd.to_numeric(
            events.get("market_count", pd.Series(index=events.index)), errors="coerce"
        )
        event_context["enable_order_book"] = _bool_series(
            events.get("enableOrderBook", pd.Series(False, index=events.index))
        )
        context = context.merge(event_context, on="event_id", how="left")

    context["topic_slug"] = context.get("topic_slug", "__missing_topic__").fillna("__missing_topic__").astype(str)
    missing_topic = context["topic_slug"].isin(["", "__missing_topic__", "nan", "<NA>"])
    if missing_topic.any():
        context.loc[missing_topic, "topic_slug"] = context.loc[missing_topic].apply(
            _infer_topic_from_market_text,
            axis=1,
        )
    for col in [
        "market_count",
        "duration_days",
        "creation_to_start_days",
        "market_neg_risk",
        "enable_order_book",
    ]:
        if col not in context.columns:
            context[col] = 0
        context[col] = pd.to_numeric(context[col], errors="coerce").fillna(0)
    context.to_parquet(context_path, index=False)

    summary = {
        "markets": int(len(markets)),
        "resolved_terminal_markets": int(len(resolutions)),
        "terminal_resolution_coverage": float(len(resolutions) / len(markets)) if len(markets) else 0.0,
        "event_context_markets": int(len(context)),
        "event_context_with_topic": int((context["topic_slug"] != "__missing_topic__").sum()),
    }
    return resolution_path, context_path, summary


def _parse_tag_slugs(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    try:
        tags = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (SyntaxError, ValueError):
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


def _infer_topic_from_market_text(row: pd.Series) -> str:
    """Infer a coarse event topic when local event tags are missing."""
    text = " ".join(
        str(row.get(col, "") or "").lower()
        for col in ["event_slug", "event_title", "question"]
    )
    keyword_groups = [
        (
            "sports",
            [
                "nba", "nfl", "mlb", "nhl", "soccer", "football", "epl", "uefa",
                "champions-league", "ufc", "tennis", "golf", "baseball",
                "basketball", "hockey", "super-bowl", "world-cup", "olympic",
                "cricket", "f1", "formula-1", "nascar", "mma", "boxing",
                "ncaa", "ncaab",
            ],
        ),
        (
            "crypto",
            [
                "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp",
                "doge", "crypto", "binance", "coinbase", "stablecoin", "defi",
                "updown", "up-or-down",
            ],
        ),
        (
            "politics_geopolitics",
            [
                "trump", "biden", "election", "president", "senate", "congress",
                "republican", "democrat", "governor", "primary", "ukraine",
                "russia", "israel", "gaza", "china", "taiwan", "iran", "war",
                "ceasefire", "supreme-court", "fed-chair", "cabinet", "nominee",
            ],
        ),
        (
            "macro_finance",
            [
                "fed", "rate", "cpi", "inflation", "recession", "gdp",
                "unemployment", "stock", "s&p", "nasdaq", "dow", "treasury",
                "oil", "gold", "earnings", "ipo", "market-cap",
            ],
        ),
        (
            "entertainment",
            [
                "oscar", "grammy", "emmy", "movie", "box-office", "taylor-swift",
                "album", "song", "billboard", "netflix", "metacritic",
                "celebrity",
            ],
        ),
        (
            "technology",
            [
                "openai", "chatgpt", "ai", "apple", "tesla", "spacex", "twitter",
                "meta", "google", "microsoft", "nvidia", "iphone", "software",
            ],
        ),
        ("weather", ["weather", "temperature", "hurricane", "snow", "rain", "storm"]),
        ("science_tail_risk", ["nuclear", "space", "moon", "mars", "earthquake", "covid", "vaccine"]),
    ]
    for topic, terms in keyword_groups:
        if any(term in text for term in terms):
            return topic
    return "other"


def _as_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _duck_exprs(cols: set[str]) -> dict[str, str]:
    def first_available(names: Iterable[str], default: str | None = None) -> str:
        for name in names:
            if name in cols:
                return f'"{name}"'
        if default is None:
            raise SystemExit(f"Missing required column, tried: {names}")
        return default

    ts_col = first_available(["timestamp", "datetime", "block_time", "time"])
    if "timestamp" in cols:
        ts_expr = (
            "CASE WHEN typeof(\"timestamp\") IN ('BIGINT','INTEGER','UBIGINT','UINTEGER','DOUBLE','FLOAT') "
            "THEN to_timestamp(\"timestamp\") ELSE CAST(\"timestamp\" AS TIMESTAMP) END"
        )
    else:
        ts_expr = f"CAST({ts_col} AS TIMESTAMP)"

    amount_col = first_available(["token_amount", "amount", "size", "shares"])
    price_col = first_available(["price"])
    outcome_col = first_available(["answer", "outcome", "token_outcome"], "NULL")
    nonusdc_col = first_available(["nonusdc_side"], "NULL")
    direction_expr = (
        "upper(CAST(\"direction\" AS VARCHAR))"
        if "direction" in cols
        else f"CASE WHEN CAST({amount_col} AS DOUBLE) >= 0 THEN 'BUY' ELSE 'SELL' END"
    )
    return {
        "address": first_available(["user", "address", "wallet"]),
        "market_id": first_available(["condition_id", "market_id"]),
        "transaction_id": first_available(["transaction_hash", "transaction_id", "hash"], "md5(random()::VARCHAR)"),
        "outcome": outcome_col,
        "nonusdc_side": f"CAST({nonusdc_col} AS VARCHAR)",
        "amount": f"abs(CAST({amount_col} AS DOUBLE))",
        "signed_amount": f"CAST({amount_col} AS DOUBLE)",
        "price": f"CAST({price_col} AS DOUBLE)",
        "timestamp": ts_expr,
        "side": direction_expr,
    }


def build_wallet_panel(
    users_path: Path,
    resolution_path: Path,
    context_path: Path,
    out_dir: Path,
    cfg: RunConfig,
    force: bool,
) -> tuple[Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "sii_full_wallet_panel.parquet"
    summary_path = out_dir / "sii_full_wallet_panel_summary.json"
    if panel_path.exists() and summary_path.exists() and not force:
        return panel_path, json.loads(summary_path.read_text())

    con = duckdb.connect(str(out_dir / "sii_full_step1.duckdb"))
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='24GB'")
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{users_path}')").fetchdf()
    cols = set(schema["column_name"].astype(str))
    e = _duck_exprs(cols)

    con.execute(
        f"""
        CREATE OR REPLACE VIEW market_context AS
        SELECT *
        FROM read_parquet('{context_path}')
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW tx0 AS
        SELECT
            CAST({e['transaction_id']} AS VARCHAR) AS transaction_id,
            CAST({e['address']} AS VARCHAR) AS address,
            CAST({e['market_id']} AS VARCHAR) AS market_id,
            {e['side']} AS side,
            COALESCE(
              CAST({e['outcome']} AS VARCHAR),
              CASE
                WHEN lower({e['nonusdc_side']}) = 'token1' THEN CAST(m.answer1 AS VARCHAR)
                WHEN lower({e['nonusdc_side']}) = 'token2' THEN CAST(m.answer2 AS VARCHAR)
                ELSE NULL
              END
            ) AS outcome,
            {e['amount']} AS amount,
            {e['price']} AS price,
            CAST({e['timestamp']} AS TIMESTAMP) AS ts
        FROM read_parquet('{users_path}')
        LEFT JOIN market_context m
          ON CAST({e['market_id']} AS VARCHAR) = CAST(m.market_id AS VARCHAR)
        WHERE {e['address']} IS NOT NULL
          AND {e['market_id']} IS NOT NULL
          AND {e['price']} BETWEEN 0 AND 1
          AND {e['amount']} > 0
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE wallet_stats AS
        SELECT
            address,
            count(*) AS trade_count,
            sum(amount * price) AS total_volume,
            min(ts) AS first_trade,
            max(ts) AS last_trade
        FROM tx0
        GROUP BY address
        HAVING trade_count >= {cfg.min_trades}
           AND total_volume >= {cfg.min_volume}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE tx AS
        SELECT t.*
        FROM tx0 t
        JOIN wallet_stats w USING(address)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE market_bounds AS
        SELECT market_id, min(ts) AS market_start, max(ts) AS market_end,
               avg(CASE WHEN side='BUY' THEN 1.0 ELSE 0.0 END) AS market_buy_ratio
        FROM tx
        GROUP BY market_id
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE obs AS
        SELECT
            t.*,
            w.first_trade,
            w.last_trade,
            (t.amount * t.price) AS notional,
            b.market_start,
            b.market_end,
            b.market_buy_ratio,
            r.resolution,
            CASE
              WHEN r.resolution IS NULL THEN NULL
              WHEN side='BUY' AND outcome = r.resolution THEN amount * (1 - price)
              WHEN side='BUY' THEN -amount * price
              WHEN side='SELL' AND outcome = r.resolution THEN -amount * (1 - price)
              ELSE amount * price
            END AS row_pnl,
            CASE WHEN side='BUY' THEN amount * price ELSE amount * (1 - price) END AS invested
        FROM tx t
        JOIN wallet_stats w USING(address)
        LEFT JOIN market_bounds b USING(market_id)
        LEFT JOIN read_parquet('{resolution_path}') r USING(market_id)
        WHERE t.ts <= w.first_trade + INTERVAL {cfg.observation_days} DAY
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE future AS
        SELECT
            t.address,
            t.market_id,
            t.outcome,
            t.ts,
            (t.amount * t.price) AS invested,
            CASE WHEN t.outcome = r.resolution THEN t.amount * (1 - t.price) ELSE -t.amount * t.price END AS future_net_profit,
            c.event_id,
            c.event_title
        FROM tx t
        JOIN wallet_stats w USING(address)
        JOIN read_parquet('{resolution_path}') r USING(market_id)
        LEFT JOIN read_parquet('{context_path}') c USING(market_id)
        WHERE t.side='BUY'
          AND t.ts > w.first_trade + INTERVAL {cfg.observation_days} DAY
          AND t.ts <= w.first_trade + INTERVAL {cfg.observation_days + cfg.horizon_days} DAY
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE base_features AS
        WITH obs_aug AS (
          SELECT *,
            CASE WHEN row_pnl > 0 THEN row_pnl ELSE NULL END AS win_pnl,
            CASE WHEN row_pnl < 0 THEN -row_pnl ELSE NULL END AS loss_pnl,
            CASE WHEN row_pnl > 0 THEN 1 ELSE 0 END AS is_win,
            CASE WHEN row_pnl IS NOT NULL THEN 1 ELSE 0 END AS is_resolved,
            CASE
              WHEN epoch(market_end) > epoch(market_start)
              THEN (epoch(ts) - epoch(market_start)) / greatest(epoch(market_end) - epoch(market_start), 1)
              ELSE 0
            END AS entry_ratio
          FROM obs
        )
        SELECT
          address,
          min(first_trade) AS first_trade_date,
          count(*) AS obs_trade_count,
          count(DISTINCT market_id) AS obs_market_count,
          sum(notional) AS obs_notional,
          coalesce(sum(row_pnl) / nullif(sum(invested), 0), 0) AS total_roi,
          least(coalesce(min(row_pnl / nullif(invested, 0)), 0), 0) AS max_drawdown,
          coalesce(sum(is_win)::DOUBLE / nullif(sum(is_resolved), 0), 0) AS win_rate,
          least(coalesce(avg(win_pnl) / nullif(avg(loss_pnl), 0), 0), 10) AS profit_loss_ratio,
          1.0 - avg(entry_ratio) AS early_entry_score,
          avg(abs(CASE WHEN side='BUY' THEN 1.0 ELSE 0.0 END - market_buy_ratio)) AS contrarian_score,
          coalesce(avg(price) / nullif(stddev_samp(price), 0), 0) AS information_ratio,
          count(DISTINCT market_id)::DOUBLE / nullif(count(*), 0) AS cross_market_diversification,
          coalesce((epoch(max(ts)) - epoch(min(ts))) / 86400.0 / nullif(count(*) - 1, 0), 0) AS avg_holding_period,
          count(*) / greatest((epoch(max(ts)) - epoch(min(ts))) / 86400.0, 1.0/24.0) AS trading_frequency
        FROM obs_aug
        GROUP BY address
        """
    )
    total_obs_notional = con.execute("SELECT sum(obs_notional) FROM base_features").fetchone()[0] or 1.0
    con.execute(
        f"""
        CREATE OR REPLACE TABLE live_features AS
        SELECT *,
          obs_notional / {total_obs_notional} AS capital_flow_centrality,
          greatest(least(total_roi / greatest(abs(max_drawdown), 0.0001), 100), -100) AS Risk_Adjusted_Return
        FROM base_features
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE event_features AS
        WITH enriched AS (
          SELECT o.address, o.market_id, o.notional,
                 coalesce(c.topic_slug, '__missing_topic__') AS topic_slug,
                 coalesce(c.market_count, 0) AS market_count,
                 coalesce(c.duration_days, 0) AS duration_days,
                 coalesce(c.creation_to_start_days, 0) AS creation_to_start_days,
                 coalesce(c.market_neg_risk, 0) AS market_neg_risk,
                 coalesce(c.enable_order_book, 0) AS enable_order_book
          FROM obs o
          LEFT JOIN read_parquet('{context_path}') c USING(market_id)
        ),
        topic_notional AS (
          SELECT address, topic_slug, sum(notional) AS topic_notional
          FROM enriched
          GROUP BY address, topic_slug
        ),
        topic_agg AS (
          SELECT address,
                 count(*) AS unique_topics,
                 sum(topic_notional) AS total_topic_notional,
                 max(topic_notional) AS max_topic_notional,
                 sum(topic_notional * topic_notional) AS topic_notional_sq_sum
          FROM topic_notional
          GROUP BY address
        ),
        static_agg AS (
          SELECT address,
                 avg(market_count) AS avg_event_market_count,
                 avg(duration_days) AS avg_event_duration_days,
                 avg(creation_to_start_days) AS avg_creation_to_start_days,
                 avg(CASE WHEN market_neg_risk IN (1, true) THEN 1.0 ELSE 0.0 END) AS neg_risk_trade_share,
                 avg(CASE WHEN enable_order_book IN (1, true) THEN 1.0 ELSE 0.0 END) AS order_book_trade_share,
                 count(*) AS n_obs
          FROM enriched
          GROUP BY address
        )
        SELECT
          s.address,
          coalesce(t.unique_topics, 0) AS unique_topics,
          coalesce(t.unique_topics::DOUBLE / nullif(s.n_obs, 0), 0) AS topic_diversification,
          coalesce(t.topic_notional_sq_sum / nullif(t.total_topic_notional * t.total_topic_notional, 0), 0) AS topic_notional_hhi,
          coalesce(t.max_topic_notional / nullif(t.total_topic_notional, 0), 0) AS top_topic_notional_share,
          s.avg_event_market_count,
          s.avg_event_duration_days,
          s.avg_creation_to_start_days,
          s.neg_risk_trade_share,
          s.order_book_trade_share
        FROM static_agg s
        LEFT JOIN topic_agg t USING(address)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE labels AS
        SELECT
          address,
          count(*) AS future_n_resolved_trades,
          sum(invested) AS future_invested,
          sum(future_net_profit) AS future_net_profit,
          coalesce(sum(future_net_profit) / nullif(sum(invested), 0), 0) AS future_roi
        FROM future
        GROUP BY address
        HAVING future_n_resolved_trades >= 1
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT lf.*, ef.* EXCLUDE(address), lab.future_n_resolved_trades,
                 lab.future_invested, lab.future_net_profit, lab.future_roi
          FROM live_features lf
          JOIN labels lab USING(address)
          LEFT JOIN event_features ef USING(address)
        ) TO '{panel_path}' (FORMAT PARQUET)
        """
    )
    summary = {
        "raw_schema": schema.to_dict(orient="records"),
        "active_wallets": int(con.execute("SELECT count(*) FROM wallet_stats").fetchone()[0]),
        "active_transactions": int(con.execute("SELECT count(*) FROM tx").fetchone()[0]),
        "observation_transactions": int(con.execute("SELECT count(*) FROM obs").fetchone()[0]),
        "future_label_transactions": int(con.execute("SELECT count(*) FROM future").fetchone()[0]),
        "panel_wallets": int(con.execute(f"SELECT count(*) FROM read_parquet('{panel_path}')").fetchone()[0]),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    con.close()
    return panel_path, summary


def add_splits_and_labels(panel: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    df = panel.copy()
    df["first_trade_date"] = pd.to_datetime(df["first_trade_date"], errors="coerce")
    df = df.dropna(subset=["first_trade_date"]).sort_values("first_trade_date").reset_index(drop=True)
    tmin, tmax = df["first_trade_date"].min(), df["first_trade_date"].max()
    span = tmax - tmin
    train_cutoff = tmin + span * 0.70
    val_cutoff = tmin + span * 0.85
    df["split"] = "test"
    df.loc[df["first_trade_date"] < train_cutoff, "split"] = "train"
    df.loc[(df["first_trade_date"] >= train_cutoff) & (df["first_trade_date"] < val_cutoff), "split"] = "val"
    threshold = df.loc[df["split"].eq("train"), "future_roi"].quantile(1 - cfg.top_label_percentile)
    df["Trader_Success_Rate"] = (df["future_roi"] >= threshold).astype(int)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def _model(random_state: int = 42):
    if LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=240,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            verbose=-1,
        )
    return RandomForestClassifier(n_estimators=250, min_samples_leaf=5, n_jobs=-1, random_state=random_state)


def _safe_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score)) if y_true.nunique() > 1 else math.nan


def _safe_ap(y_true: pd.Series, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score)) if y_true.nunique() > 1 else math.nan


def evaluate_regime(df: pd.DataFrame, name: str, cols: list[str], cfg: RunConfig) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [c for c in cols if c in df.columns]
    folds = np.array_split(df.sort_values("first_trade_date").index.to_numpy(), cfg.rolling_folds + 1)
    fold_rows = []
    selected_rows = []
    prediction_rows = []
    importance_rows = []
    for i in range(1, len(folds)):
        train_idx = np.concatenate(folds[:i])
        test_idx = folds[i]
        train = df.loc[train_idx]
        test = df.loc[test_idx]
        if train["Trader_Success_Rate"].nunique() < 2 or test.empty:
            continue
        model = _model(42 + i)
        model.fit(train[cols], train["Trader_Success_Rate"])
        if hasattr(model, "feature_importances_"):
            for feature, importance in zip(cols, model.feature_importances_):
                importance_rows.append(
                    {
                        "regime": name,
                        "fold": i,
                        "feature": feature,
                        "importance": float(importance),
                    }
                )
        scores = model.predict_proba(test[cols])[:, 1]
        ranked = test.copy()
        ranked["score"] = scores
        ranked["fold"] = i
        ranked["regime"] = name
        prediction_rows.append(
            ranked[
                [
                    "regime",
                    "fold",
                    "address",
                    "score",
                    "Trader_Success_Rate",
                    "future_net_profit",
                    "future_invested",
                    "future_roi",
                    "future_n_resolved_trades",
                ]
            ]
        )
        k = max(1, int(math.ceil(len(ranked) * cfg.top_select_percentile)))
        selected = ranked.nlargest(k, "score").copy()
        selected_rows.append(
            selected[
                [
                    "regime",
                    "fold",
                    "address",
                    "score",
                    "Trader_Success_Rate",
                    "future_net_profit",
                    "future_invested",
                    "future_roi",
                    "future_n_resolved_trades",
                ]
            ]
        )
        fold_rows.append(
            {
                "regime": name,
                "fold": i,
                "test_wallets": int(len(test)),
                "test_positives": int(test["Trader_Success_Rate"].sum()),
                "auc": _safe_auc(test["Trader_Success_Rate"], scores),
                "average_precision": _safe_ap(test["Trader_Success_Rate"], scores),
                "precision_at_20": float(ranked.nlargest(min(20, len(ranked)), "score")["Trader_Success_Rate"].mean()),
                "top_10_percent_precision": float(selected["Trader_Success_Rate"].mean()),
                "selected_wallets": int(k),
                "selected_future_net_profit": float(selected["future_net_profit"].sum()),
                "selected_mean_roi": float(selected["future_roi"].mean()),
            }
        )
    folds_df = pd.DataFrame(fold_rows)
    selected_df = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    prediction_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    importance_df = pd.DataFrame(importance_rows)

    rng = np.random.default_rng(42)
    random_profits = []
    for _ in range(cfg.random_seeds):
        profit = 0.0
        roi_values = []
        for _, row in folds_df.iterrows():
            fold = int(row["fold"])
            fold_idx = folds[fold]
            pool = df.loc[fold_idx]
            k = int(row["selected_wallets"])
            if len(pool) == 0:
                continue
            picks = pool.iloc[rng.choice(len(pool), size=min(k, len(pool)), replace=False)]
            profit += float(picks["future_net_profit"].sum())
            roi_values.extend(picks["future_roi"].tolist())
        random_profits.append({"random_profit": profit, "random_mean_roi": float(np.mean(roi_values)) if roi_values else 0.0})
    random_df = pd.DataFrame(random_profits)
    total_profit = float(folds_df["selected_future_net_profit"].sum()) if not folds_df.empty else 0.0
    mean_roi = float(folds_df["selected_mean_roi"].mean()) if not folds_df.empty else 0.0
    summary = {
        "regime": name,
        "features": cols,
        "wallet_rows": int(len(df)),
        "folds": int(len(folds_df)),
        "rolling_auc": float(folds_df["auc"].mean()) if not folds_df.empty else math.nan,
        "rolling_average_precision": float(folds_df["average_precision"].mean()) if not folds_df.empty else math.nan,
        "precision_at_20": float(folds_df["precision_at_20"].mean()) if not folds_df.empty else math.nan,
        "top_10_percent_precision": float(folds_df["top_10_percent_precision"].mean()) if not folds_df.empty else math.nan,
        "walk_forward_total_future_net_profit": total_profit,
        "walk_forward_mean_selected_roi": mean_roi,
        "random_profit_percentile": float((random_df["random_profit"] < total_profit).mean()) if not random_df.empty else math.nan,
        "random_roi_percentile": float((random_df["random_mean_roi"] < mean_roi).mean()) if not random_df.empty else math.nan,
        "random_profit_p025": float(random_df["random_profit"].quantile(0.025)) if not random_df.empty else math.nan,
        "random_profit_p975": float(random_df["random_profit"].quantile(0.975)) if not random_df.empty else math.nan,
    }
    return summary, folds_df, selected_df, prediction_df, importance_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users-parquet", default="data/external/sii_polymarket_data/users.parquet")
    parser.add_argument("--markets-parquet", default="data/external/sii_polymarket_data/markets.parquet")
    parser.add_argument("--events-csv", default="polymarket_events.csv")
    parser.add_argument("--output-dir", default="results/new_data_sources/full_data_step1")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--force-panel", action="store_true")
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--min-volume", type=float, default=1000.0)
    parser.add_argument("--rolling-folds", type=int, default=8)
    parser.add_argument("--random-seeds", type=int, default=200)
    args = parser.parse_args()

    users_path = Path(args.users_parquet)
    markets_path = Path(args.markets_parquet)
    events_csv = Path(args.events_csv)
    out_dir = Path(args.output_dir)
    if args.download_missing:
        users_path = _download_missing(users_path, "users.parquet")
        markets_path = _download_missing(markets_path, "markets.parquet")

    cfg = RunConfig(
        min_trades=args.min_trades,
        min_volume=args.min_volume,
        observation_days=1,
        horizon_days=7,
        top_label_percentile=0.20,
        rolling_folds=args.rolling_folds,
        random_seeds=args.random_seeds,
        top_select_percentile=0.10,
    )

    resolution_path, context_path, market_summary = build_market_context(markets_path, events_csv, out_dir)
    panel_path, panel_summary = build_wallet_panel(users_path, resolution_path, context_path, out_dir, cfg, args.force_panel)
    df = add_splits_and_labels(pd.read_parquet(panel_path), cfg)
    df.to_csv(out_dir / "sii_full_model_input.csv", index=False)

    summaries = []
    all_folds = []
    all_selected = []
    all_predictions = []
    all_importance = []
    for name, cols in FEATURE_REGIMES.items():
        summary, folds, selected, predictions, importance = evaluate_regime(df, name, cols, cfg)
        summaries.append(summary)
        all_folds.append(folds)
        all_selected.append(selected)
        all_predictions.append(predictions)
        all_importance.append(importance)

    summary_df = pd.DataFrame(summaries)
    folds_df = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    selected_df = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    importance_df = pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()
    summary_df.to_csv(out_dir / "full_data_step1_summary.csv", index=False)
    folds_df.to_csv(out_dir / "full_data_step1_folds.csv", index=False)
    selected_df.to_csv(out_dir / "full_data_step1_selected_wallets.csv", index=False)
    predictions_df.to_csv(out_dir / "full_data_step1_predictions.csv", index=False)
    importance_df.to_csv(out_dir / "full_data_step1_feature_importance.csv", index=False)
    report = {
        "config": cfg.__dict__,
        "market_summary": market_summary,
        "panel_summary": panel_summary,
        "split_counts": df["split"].value_counts().to_dict(),
        "label_counts": df["Trader_Success_Rate"].value_counts().to_dict(),
        "summary": summaries,
    }
    (out_dir / "full_data_step1_summary.json").write_text(json.dumps(report, indent=2, default=str))
    print(summary_df.to_string(index=False))
    print(f"Wrote Step 1 full-data outputs to {out_dir}")


if __name__ == "__main__":
    main()
