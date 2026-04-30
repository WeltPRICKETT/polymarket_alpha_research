"""
Author: AI Assistant
Date: 2026-04-11
Description: Event Study Engine
Analyzes price dynamics (Abnormal Returns / Price Delta) immediately
following trades executed by identified "Informed Traders".
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class EventStudyEngine:
    def __init__(self, informed_csv_path=None, transactions_csv_path=None):
        self.informed_path = informed_csv_path or (PROJECT_ROOT / "results" / "informed_traders.csv")
        self.tx_path = transactions_csv_path or (PROJECT_ROOT / "data" / "processed" / "cleaned_transactions.csv")
        
        self.informed_df = None
        self.tx_df = None
        
    def load_data(self):
        logger.info("Loading Event Study data...")
        self.informed_df = pd.read_csv(self.informed_path)
        
        # We only care about positive predictions (the ones we would follow)
        if "predicted_label" in self.informed_df.columns:
            self.informed_addresses = set(self.informed_df[self.informed_df["predicted_label"] == 1]["address"])
        else:
            self.informed_addresses = set(self.informed_df["address"])
            
        logger.info(f"Loaded {len(self.informed_addresses)} informed addresses.")
        
        # Load tx, sort by time for price reconstruction
        self.tx_df = pd.read_csv(self.tx_path)
        self.tx_df["timestamp"] = pd.to_datetime(self.tx_df["timestamp"])
        self.tx_df.sort_values(by=["market_id", "timestamp"], inplace=True)
        logger.info(f"Loaded {len(self.tx_df):,} transactions for price tick reconstruction.")

    def run_event_study(self, time_windows_min=[5, 15, 60, 1440]):
        """
        Calculates Price Delta (CAR proxy) at t + window for every informed trade.
        """
        logger.info("Reconstructing market tick data via transaction log...")
        
        # Subset of informed trades
        informed_tx = self.tx_df[self.tx_df["address"].isin(self.informed_addresses)].copy()
        logger.info(f"Found {len(informed_tx):,} trades executed by informed traders.")
        
        if len(informed_tx) == 0:
            logger.warning("No informed trades found to analyze!")
            return pd.DataFrame()

        results = []
        
        # Group global market trades into a dict for fast local processing
        # Only keep markets traded by informed traders to save memory
        target_markets = set(informed_tx["market_id"])
        market_series = {
            m: grp.set_index("timestamp")["price"].sort_index()
            for m, grp in self.tx_df[self.tx_df["market_id"].isin(target_markets)].groupby("market_id")
        }

        # Process each informed trade
        for _, trade in informed_tx.iterrows():
            t_entry = trade["timestamp"]
            m_id = trade["market_id"]
            side = str(trade["side"]).upper()
            price_entry = trade["price"]
            
            series = market_series.get(m_id)
            if series is None or series.empty:
                continue

            # We need to find the price at exactly t + window. We will use forward fill 
            # (last traded price before or exactly at t+window), or the first trade if none exist before.
            # get_indexer with method='ffill' is efficient.
            
            record = {
                "transaction_id": trade.get("transaction_id", ""),
                "address": trade["address"],
                "market_id": m_id,
                "timestamp": t_entry,
                "side": side,
                "entry_price": price_entry,
                "volume": trade.get("volume", trade.get("amount", 0))
            }
            
            for w in time_windows_min:
                t_target = t_entry + pd.Timedelta(minutes=w)
                
                # Get the latest price <= t_target
                past_prices = series.loc[:t_target]
                if len(past_prices) > 0:
                    price_target = past_prices.iloc[-1]
                else:
                    # If very first trade in market, fallback to entry
                    price_target = price_entry 

                # Delta: if we bought YES, profit = price_target - price_entry
                # If we Sold (bought NO), profit = price_entry - price_target
                # Wait, if side=BUY at price P, we bought YES at P. profit is Pt - P0.
                # If side=SELL at price P, it usually means we sold YES (thereby shorting it).
                # Actually in Polymarket a SELL means selling a YES position we already had.
                # True front-running generally copies BUY orders (new position establishment).
                # Let's map delta relative to BUY direction.
                
                delta = price_target - price_entry
                if side == "SELL":
                    delta = -delta  # If they sold, price drop is a win
                    
                record[f"delta_{w}m"] = delta
                record[f"price_{w}m"] = price_target
                
            results.append(record)

        res_df = pd.DataFrame(results)
        logger.info("Event study metrics aggregated.")
        return res_df

    def evaluate_and_save(self, res_df: pd.DataFrame):
        if res_df.empty:
            return
            
        out_dir = PROJECT_ROOT / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Aggregate stats
        summary = {}
        delta_cols = [c for c in res_df.columns if c.startswith("delta_")]
        
        # Plot styling
        plt.rcParams.update({
            'axes.facecolor': '#ffffff', 'figure.facecolor': '#ffffff',
            'axes.edgecolor': '#000000', 'grid.color': '#dddddd', 'grid.alpha': 0.5,
            'grid.linestyle': '--', 'axes.linewidth': 1.0,
            'text.color': '#000000', 'axes.labelcolor': '#000000', 
            'xtick.color': '#000000', 'ytick.color': '#000000',
            'font.family': 'serif', 'font.size': 11
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = []
        means = []
        errs = []
        
        logger.info("=== EVENT STUDY ABNORMAL RETURNS ===")
        for col in sorted(delta_cols, key=lambda x: int(x.split('_')[1][:-1])):
            win_min = int(col.split('_')[1][:-1])
            mu = res_df[col].mean()
            std = res_df[col].std()
            sem = std / np.sqrt(len(res_df)) if len(res_df)> 1 else 0
            
            times.append(str(win_min))
            means.append(mu)
            errs.append(sem * 1.96) # 95% CI
            
            summary[f"Mean_Delta_{win_min}m"] = round(mu, 4)
            summary[f"Win_Rate_{win_min}m"] = round((res_df[col] > 0).mean(), 4)
            logger.info(f"+{win_min}m: Avg Delta {mu:+.4f} | Profitable {summary[f'Win_Rate_{win_min}m']:.1%}")

        ax.errorbar(times, means, yerr=errs, fmt='-o', color='#1f77b4', markersize=6, 
                    ecolor='#000000', capsize=5, capthick=1, linewidth=2)
        ax.axhline(0, color='grey', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.set_title("Cumulative Abnormal Returns (Delta Price)\nAfter Informed Trade Executions", pad=15)
        ax.set_xlabel("Time Horizon (Minutes)")
        ax.set_ylabel("Average Price Deviation (cents)")
        ax.grid(True)
        
        plot_path = PROJECT_ROOT / "results" / "plots" / "event_study_trajectory.png"
        plot_path.parent.mkdir(exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved Event Study Plot to {plot_path.name}")
        
        # Save raw data
        res_df.to_csv(out_dir / "event_study_results.csv", index=False)
        logger.info("Saved raw event study results to event_study_results.csv")
        
        # Basic correlation of size vs impact
        if "volume" in res_df.columns:
            largest = res_df[res_df["volume"] >= 500]
            if len(largest) > 0:
                logger.info("--- Whale Size Impacts (Vol >= $500) ---")
                for col in delta_cols:
                    mu = largest[col].mean()
                    logger.info(f" +{col.split('_')[1]}: {mu:+.4f}")
        
        return summary

if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    engine = EventStudyEngine()
    engine.load_data()
    df = engine.run_event_study(time_windows_min=[1, 5, 15, 30, 60, 240, 1440])
    engine.evaluate_and_save(df)
