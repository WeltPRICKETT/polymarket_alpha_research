import polars as pl

p = "processed/trades_window_20260525_20260527_exclusive.csv"
df = pl.scan_csv(
    p,
    schema_overrides={"market_id": pl.Utf8, "maker": pl.Utf8, "taker": pl.Utf8},
).with_columns(pl.col("timestamp").str.slice(0, 10).alias("date"))

summary = df.select(
    [
        pl.len().alias("rows"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
        pl.col("market_id").n_unique().alias("unique_markets"),
        pl.col("maker").n_unique().alias("unique_makers"),
        pl.col("taker").n_unique().alias("unique_takers"),
        pl.col("usd_amount").sum().alias("usd_notional"),
        pl.col("price").min().alias("min_price"),
        pl.col("price").max().alias("max_price"),
    ]
).collect()
print(summary)

by_day = (
    df.group_by("date")
    .agg(
        [
            pl.len().alias("rows"),
            pl.col("market_id").n_unique().alias("markets"),
            pl.col("usd_amount").sum().alias("usd_notional"),
        ]
    )
    .sort("date")
    .collect()
)
print(by_day)
