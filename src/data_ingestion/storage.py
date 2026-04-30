"""
Author: AI Assistant
Date: 2026-03-26
Description: Optimized data storage with batch inserts and rich query methods.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, text, func
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config.settings import DATABASE_URL
from loguru import logger
import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

Base = declarative_base()

class Transaction(Base):
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, unique=True, index=True)
    address = Column(String, index=True)
    market_id = Column(String, index=True)
    side = Column(String)       # BUY / SELL
    outcome = Column(String, nullable=True)  # YES / NO / Up / Down / etc.
    amount = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime)

class Position(Base):
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String, index=True)
    market_id = Column(String, index=True)
    token_balance = Column(Float)
    avg_price = Column(Float)
    last_updated = Column(DateTime)

class Market(Base):
    __tablename__ = 'markets'
    
    market_id = Column(String, primary_key=True)
    question = Column(String)
    resolved = Column(Boolean, default=False)
    resolution_outcome = Column(String, nullable=True)

class Storage:
    """Handles database connections and CRUD operations (optimized)."""
    
    def __init__(self, db_url: str = DATABASE_URL):
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Connected to database: {db_url}")

    # ── Batch Insert (10-50x faster than row-by-row) ─────────────────────────

    def save_transactions(self, records: List[Dict[str, Any]]):
        """
        Save transaction records using batch insert with pre-deduplication.
        """
        if not records:
            return

        session = self.Session()
        try:
            # 1. Load existing tx_ids in one query for dedup
            existing_ids = self.get_existing_tx_hashes()

            # 2. Parse timestamps and filter duplicates
            new_mappings = []
            for rec in records:
                tx_id = rec.get("transaction_id")
                if tx_id in existing_ids:
                    continue
                existing_ids.add(tx_id)

                ts = rec.get("timestamp")
                if isinstance(ts, str):
                    try:
                        ts = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        ts = ts.replace(tzinfo=None)
                    except ValueError:
                        ts = None

                new_mappings.append({
                    "transaction_id": tx_id,
                    "address": rec.get("address"),
                    "market_id": rec.get("market_id"),
                    "side": rec.get("side"),
                    "outcome": rec.get("outcome") or rec.get("_outcome"),  # P0.1: retain outcome
                    "amount": rec.get("amount"),
                    "price": rec.get("price"),
                    "timestamp": ts,
                })

            # 3. Batch insert in chunks of 500
            if new_mappings:
                chunk_size = 500
                for i in range(0, len(new_mappings), chunk_size):
                    chunk = new_mappings[i:i + chunk_size]
                    session.bulk_insert_mappings(Transaction, chunk)
                session.commit()
                logger.info(f"Batch inserted {len(new_mappings)} new records (skipped {len(records) - len(new_mappings)} duplicates).")
            else:
                logger.info(f"No new records to insert (all {len(records)} were duplicates).")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save transactions: {e}")
            raise
        finally:
            session.close()

    def load_transactions_df(self) -> pd.DataFrame:
        """Load all transactions into a Pandas DataFrame."""
        query = "SELECT * FROM transactions"
        return pd.read_sql(query, self.engine)

    def clear_all_transactions(self):
        """Delete all rows from the transactions table."""
        session = self.Session()
        try:
            session.query(Transaction).delete()
            session.commit()
            logger.info("Cleared all transactions from the database.")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to clear transactions: {e}")
        finally:
            session.close()

    def get_existing_tx_hashes(self) -> set:
        """Return a set of all transaction_id values already in the DB (for dedup)."""
        session = self.Session()
        try:
            rows = session.query(Transaction.transaction_id).all()
            return {r[0] for r in rows}
        finally:
            session.close()

    def get_total_count(self) -> int:
        """Return the total number of transactions in the DB."""
        session = self.Session()
        try:
            return session.query(Transaction).count()
        finally:
            session.close()

    # ── Rich Query Methods (for API endpoints) ──────────────────────────────

    def get_wallet_stats(self, limit: int = 20) -> List[Dict]:
        """Return top wallets ranked by total volume."""
        query = text("""
            SELECT address,
                   COUNT(*) as trade_count,
                   ROUND(SUM(amount * price), 2) as total_volume,
                   COUNT(DISTINCT market_id) as markets_traded,
                   MIN(timestamp) as first_trade,
                   MAX(timestamp) as last_trade
            FROM transactions
            GROUP BY address
            ORDER BY total_volume DESC
            LIMIT :limit
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"limit": limit}).fetchall()
            return [
                {
                    "address": r[0],
                    "trade_count": r[1],
                    "total_volume": r[2],
                    "markets_traded": r[3],
                    "first_trade": str(r[4]) if r[4] else None,
                    "last_trade": str(r[5]) if r[5] else None,
                }
                for r in rows
            ]

    def get_trade_timeline(self, interval: str = "day") -> List[Dict]:
        """Return trade count and volume aggregated by time interval."""
        # SQLite date formatting
        if interval == "hour":
            date_fmt = "%Y-%m-%d %H:00:00"
        elif interval == "day":
            date_fmt = "%Y-%m-%d"
        else:
            date_fmt = "%Y-%m"

        query = text(f"""
            SELECT strftime('{date_fmt}', timestamp) as period,
                   COUNT(*) as trade_count,
                   ROUND(SUM(amount * price), 2) as volume,
                   COUNT(DISTINCT address) as unique_wallets
            FROM transactions
            WHERE timestamp IS NOT NULL
            GROUP BY period
            ORDER BY period
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(query).fetchall()
            return [
                {
                    "period": r[0],
                    "trade_count": r[1],
                    "volume": r[2],
                    "unique_wallets": r[3],
                }
                for r in rows
            ]

    def get_market_overview(self, limit: int = 20) -> List[Dict]:
        """Return market-level aggregates."""
        query = text("""
            SELECT market_id,
                   COUNT(*) as trade_count,
                   ROUND(SUM(amount * price), 2) as total_volume,
                   COUNT(DISTINCT address) as unique_traders,
                   MIN(timestamp) as first_trade,
                   MAX(timestamp) as last_trade
            FROM transactions
            GROUP BY market_id
            ORDER BY total_volume DESC
            LIMIT :limit
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"limit": limit}).fetchall()
            return [
                {
                    "market_id": r[0],
                    "trade_count": r[1],
                    "total_volume": r[2],
                    "unique_traders": r[3],
                    "first_trade": str(r[4]) if r[4] else None,
                    "last_trade": str(r[5]) if r[5] else None,
                }
                for r in rows
            ]

    def get_latest_timestamp(self) -> Optional[str]:
        """Return the latest transaction timestamp in the DB."""
        session = self.Session()
        try:
            result = session.query(func.max(Transaction.timestamp)).scalar()
            return str(result) if result else None
        finally:
            session.close()

    def get_unique_wallet_count(self) -> int:
        """Return the number of unique wallet addresses."""
        session = self.Session()
        try:
            return session.query(func.count(func.distinct(Transaction.address))).scalar() or 0
        finally:
            session.close()

    def get_unique_market_count(self) -> int:
        """Return the number of unique market IDs."""
        session = self.Session()
        try:
            return session.query(func.count(func.distinct(Transaction.market_id))).scalar() or 0
        finally:
            session.close()
