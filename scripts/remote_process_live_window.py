import os
import sys
from datetime import datetime, timezone

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poly_utils.utils import get_lean_markets


ORDERS_CSV = "data/orderFilled.csv"
START_DATE = os.environ.get("PROCESS_START_DATE", "2026-05-25")
END_DATE = os.environ.get("PROCESS_END_DATE", "2026-05-27")
OUT = os.environ.get(
    "PROCESS_WINDOW_OUT",
    f"processed/trades_window_{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}_exclusive.csv",
)


def epoch(date_str: str) -> int:
    return int(datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc).timestamp())


def main() -> None:
    start_ts = epoch(START_DATE)
    end_ts = epoch(END_DATE)
    print("window", START_DATE, END_DATE, "epoch", start_ts, end_ts, "out", OUT, flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

    markets_df = get_lean_markets().rename({"id": "market_id"})
    markets_long = markets_df.select(["market_id", "token1", "token2"]).melt(
        id_vars="market_id",
        value_vars=["token1", "token2"],
        variable_name="side",
        value_name="asset_id",
    )
    markets_long = markets_long.filter(pl.col("asset_id").is_not_null()).unique(subset=["asset_id"])
    print("markets", len(markets_df), "tokens", len(markets_long), flush=True)

    raw = (
        pl.scan_csv(
            ORDERS_CSV,
            schema_overrides={"makerAssetId": pl.Utf8, "takerAssetId": pl.Utf8},
        )
        .filter((pl.col("timestamp") >= start_ts) & (pl.col("timestamp") < end_ts))
        .with_columns(
            pl.when(pl.col("makerAssetId") != "0")
            .then(pl.col("makerAssetId"))
            .otherwise(pl.col("takerAssetId"))
            .alias("nonusdc_asset_id")
        )
    )

    total_rows = raw.select(pl.len().alias("rows")).collect().item()
    print("raw_window_rows", total_rows, flush=True)

    joined = raw.join(
        markets_long.lazy(),
        left_on="nonusdc_asset_id",
        right_on="asset_id",
        how="left",
    )
    known_rows = joined.select(pl.col("market_id").is_not_null().sum().alias("known")).collect().item()
    print(
        "known_rows",
        known_rows,
        "row_coverage",
        known_rows / total_rows if total_rows else 0,
        flush=True,
    )

    trades = (
        joined.filter(pl.col("market_id").is_not_null())
        .with_columns(
            [
                pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("timestamp"),
                pl.when(pl.col("makerAssetId") == "0")
                .then(pl.lit("USDC"))
                .otherwise(pl.col("side"))
                .alias("makerAsset"),
                pl.when(pl.col("takerAssetId") == "0")
                .then(pl.lit("USDC"))
                .otherwise(pl.col("side"))
                .alias("takerAsset"),
            ]
        )
        .with_columns(
            [
                (pl.col("makerAmountFilled") / 10**6).alias("makerAmountFilled"),
                (pl.col("takerAmountFilled") / 10**6).alias("takerAmountFilled"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("takerAsset") == "USDC")
                .then(pl.lit("BUY"))
                .otherwise(pl.lit("SELL"))
                .alias("taker_direction"),
                pl.when(pl.col("takerAsset") == "USDC")
                .then(pl.lit("SELL"))
                .otherwise(pl.lit("BUY"))
                .alias("maker_direction"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("makerAssetId") == "0")
                .then(pl.col("takerAsset"))
                .otherwise(pl.col("makerAsset"))
                .alias("nonusdc_side"),
                pl.when(pl.col("takerAsset") == "USDC")
                .then(pl.col("takerAmountFilled"))
                .otherwise(pl.col("makerAmountFilled"))
                .alias("usd_amount"),
                pl.when(pl.col("takerAsset") != "USDC")
                .then(pl.col("takerAmountFilled"))
                .otherwise(pl.col("makerAmountFilled"))
                .alias("token_amount"),
                pl.when(pl.col("takerAsset") == "USDC")
                .then(pl.col("takerAmountFilled") / pl.col("makerAmountFilled"))
                .otherwise(pl.col("makerAmountFilled") / pl.col("takerAmountFilled"))
                .cast(pl.Float64)
                .alias("price"),
            ]
        )
        .select(
            [
                "timestamp",
                "market_id",
                "maker",
                "taker",
                "nonusdc_side",
                "maker_direction",
                "taker_direction",
                "price",
                "usd_amount",
                "token_amount",
                "transactionHash",
                "block_number",
            ]
        )
    )
    df = trades.collect()
    df.write_csv(OUT)
    print("wrote_rows", len(df), "out", OUT, flush=True)


if __name__ == "__main__":
    main()
