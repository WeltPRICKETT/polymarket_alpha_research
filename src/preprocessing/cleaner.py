"""
Author: AI Assistant
Date: 2026-03-17
Description: Data cleaning module handling missing values, anomalies, and basic filtering.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional

class DataCleaner:
    """Class responsible for cleaning raw Polymarket transaction data."""

    def __init__(self, raw_df: pd.DataFrame):
        """
        Initialize the cleaner with a raw DataFrame.
        """
        self.df = raw_df.copy()
        logger.info(f"Initialized DataCleaner with {len(self.df)} records.")

    def handle_missing_values(self):
        """Handles missing values in critical columns."""
        initial_len = len(self.df)
        
        # Drop rows where 'address' or 'market_id' is null, as these are mandatory
        self.df = self.df.dropna(subset=['address', 'market_id', 'timestamp'])
        
        # Fill missing amounts/prices with 0 if necessary
        self.df['amount'] = self.df['amount'].fillna(0.0)
        self.df['price'] = self.df['price'].fillna(0.0)
        
        dropped = initial_len - len(self.df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows due to missing critical values.")
        return self

    def remove_anomalies(self):
        """Removes extreme outliers (e.g. invalid prices > 1.0 or < 0.0)."""
        initial_len = len(self.df)
        
        # Polymarket shares are priced between 0 and 1 USDC
        self.df = self.df[(self.df['price'] >= 0.0) & (self.df['price'] <= 1.0)]
        self.df = self.df[self.df['amount'] >= 0.0]
        
        dropped = initial_len - len(self.df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} anomalous rows (invalid price/amount bounds).")
        return self

    def format_timestamps(self):
        """Standardizes timestamp format to datetime objects."""
        if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
            self.df = self.df.dropna(subset=['timestamp'])
        return self

    def filter_active_traders(self, min_trades: int = 10, min_volume: float = 1000.0, top_percentile: Optional[float] = None):
        """
        Filters the dataset to only include active addresses.
        If `top_percentile` is provided (e.g., 0.1 for top 10%), it will further restrict the pool
        to the top position holders based on total trading volume.
        """
        # Calculate volume per transaction
        self.df['volume'] = self.df['amount'] * self.df['price']
        
        # Group by address
        trader_stats = self.df.groupby('address').agg(
            trade_count=('transaction_id', 'count'),
            total_volume=('volume', 'sum')
        ).reset_index()
        
        # Initial base filter
        active_stats = trader_stats[
            (trader_stats['trade_count'] >= min_trades) & 
            (trader_stats['total_volume'] >= min_volume)
        ].copy()

        # Dynamic Whale Extraction (Top Position Holders)
        if top_percentile is not None and len(active_stats) > 0:
            threshold_volume = active_stats['total_volume'].quantile(1.0 - top_percentile)
            active_stats = active_stats[active_stats['total_volume'] >= threshold_volume]
            logger.info(f"Applied top {top_percentile*100}% whale filter. Volume threshold: {threshold_volume:.2f} USDC")

        active_addresses = active_stats['address']
        original_traders = self.df['address'].nunique()
        self.df = self.df[self.df['address'].isin(active_addresses)]
        filtered_traders = self.df['address'].nunique()
        
        logger.info(f"Filtered active/top traders. Kept {filtered_traders}/{original_traders} addresses.")
        return self

    def clean(self, min_trades: int = 10, min_volume: float = 1000.0, top_percentile: Optional[float] = None) -> pd.DataFrame:
        """Runs the full cleaning pipeline and returns the cleaned DataFrame."""
        logger.info("Starting data cleaning pipeline...")
        self.format_timestamps()
        self.handle_missing_values()
        self.remove_anomalies()
        self.filter_active_traders(min_trades=min_trades, min_volume=min_volume, top_percentile=top_percentile)
        
        # Sort chronologically
        self.df = self.df.sort_values(by=['address', 'timestamp']).reset_index(drop=True)
        
        logger.info(f"Data cleaning complete. Final record count: {len(self.df)}")
        return self.df
