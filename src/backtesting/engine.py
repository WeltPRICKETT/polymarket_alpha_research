"""
Author: AI Assistant
Date: 2026-04-11
Description: Strategy Backtesting Engine
Simulates "Copy Trading" following "Informed Traders" with a configurable latency.
Assuming a fixed portfolio size and flat size per trade.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class StrategyBacktester:
    def __init__(self, 
                 latency_minutes: int = 5,
                 trade_size: float = 100.0,
                 fees_pct: float = 0.001):
        self.latency = pd.Timedelta(minutes=latency_minutes)
        self.trade_size = trade_size
        self.fees = fees_pct
        
        self.informed_path = PROJECT_ROOT / "results" / "informed_traders.csv"
        self.tx_path = PROJECT_ROOT / "data" / "processed" / "cleaned_transactions.csv"
        self.res_path = PROJECT_ROOT / "data" / "processed" / "market_resolutions.csv"
        
    def load_data(self):
        logger.info(f"Loading Backtest data (Latency: {self.latency.total_seconds()/60:.0f}m, Trade Size: ${self.trade_size})")
        informed_df = pd.read_csv(self.informed_path)
        if "predicted_label" in informed_df.columns:
            self.experts = set(informed_df[informed_df["predicted_label"] == 1]["address"])
        else:
            self.experts = set(informed_df["address"])
            
        self.tx_df = pd.read_csv(self.tx_path)
        self.tx_df["timestamp"] = pd.to_datetime(self.tx_df["timestamp"])
        self.tx_df.sort_values(by=["market_id", "timestamp"], inplace=True)
        
        res_df = pd.read_csv(self.res_path) if self.res_path.exists() else pd.DataFrame()
        if not res_df.empty:
            def norm_res(s):
                s = str(s).lower()
                if s in ["yes", "1", "true", "up"]: return "YES"
                if s in ["no", "0", "false", "down"]: return "NO"
                return "OPEN"
            self.resolutions = {r["market_id"]: norm_res(r["resolution"]) for _, r in res_df.iterrows()}
        else:
            self.resolutions = {}

    def simulate(self):
        logger.info("Initializing Backtest simulation...")
        
        expert_tx = self.tx_df[self.tx_df["address"].isin(self.experts)].copy()
        
        target_markets = set(expert_tx["market_id"])
        
        # Build price interpolator for target markets
        market_series = {
            m: grp.set_index("timestamp")[["price", "side"]]
            for m, grp in self.tx_df[self.tx_df["market_id"].isin(target_markets)].groupby("market_id")
        }

        portfolio = []
        pnl_tracking = []
        
        balance = 10000.0  # Starting nominal balance
        peak_balance = balance
        max_drawdown = 0.0
        
        for _, trade in expert_tx.iterrows():
            t_expert = trade["timestamp"]
            m_id = trade["market_id"]
            expert_side = str(trade["side"]).upper()
            expert_price = trade["price"]
            
            # We copy BUY trades (i.e. establishing positions)
            if expert_side != "BUY":
                continue
                
            final_res = self.resolutions.get(m_id, "OPEN")
            if final_res not in ["YES", "NO"]:
                continue
                
            series = market_series.get(m_id)
            if series is None or series.empty:
                continue
                
            t_exec = t_expert + self.latency
            
            past_prices = series.loc[:t_exec]
            if len(past_prices) > 0:
                our_price = past_prices.iloc[-1]["price"]
            else:
                our_price = expert_price

            expert_bet = "YES" if expert_price >= 0.5 else "NO"
            
            if expert_bet == final_res:
                profit = (1.0 - our_price) * (self.trade_size / our_price)
            else:
                profit = -self.trade_size
                
            fee = self.trade_size * self.fees
            net_profit = profit - fee
            
            balance += net_profit
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd
                
            portfolio.append({
                "timestamp": t_exec,
                "market_id": m_id,
                "expert_address": trade["address"],
                "exec_price": our_price,
                "expert_price": expert_price,
                "slippage": our_price - expert_price if expert_bet == "YES" else expert_price - our_price,
                "net_profit": net_profit,
                "balance": balance
            })

        if not portfolio:
            logger.warning("No valid trades fulfilled during simulation.")
            return

        port_df = pd.DataFrame(portfolio).sort_values("timestamp")
        
        metrics = {
            "Total_Trades": len(port_df),
            "Win_Rate": (port_df["net_profit"] > 0).mean(),
            "Total_Net_Profit": port_df["net_profit"].sum(),
            "ROI_Pct": port_df["net_profit"].sum() / (port_df["exec_price"].mean() * len(port_df) + 1e-5),
            "Max_Drawdown": max_drawdown,
            "Average_Slippage": port_df["slippage"].mean()
        }
        
        port_df["date"] = port_df["timestamp"].dt.date
        daily_pnl = port_df.groupby("date")["net_profit"].sum()
        if len(daily_pnl) > 1 and daily_pnl.std() > 0:
            metrics["Sharpe_Ratio (Annualized)"] = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(365)
        else:
            metrics["Sharpe_Ratio (Annualized)"] = 0.0

        logger.info("\n--- BACKTEST RESULTS ---")
        for k, v in metrics.items():
            if "Pct" in k or "Rate" in k:
                logger.info(f"{k}: {v:.2%}")
            else:
                logger.info(f"{k}: {v:.4f}")
                
        out_dir = PROJECT_ROOT / "results"
        port_df.to_csv(out_dir / "backtest_trades.csv", index=False)
        with open(out_dir / "backtest_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        self._plot_equity(port_df)

    def _plot_equity(self, port_df):
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            'axes.facecolor': '#ffffff', 'figure.facecolor': '#ffffff',
            'axes.edgecolor': '#000000', 'grid.color': '#dddddd', 'grid.alpha': 0.5,
            'grid.linestyle': '--', 'axes.linewidth': 1.0,
            'text.color': '#000000', 'axes.labelcolor': '#000000', 
            'xtick.color': '#000000', 'ytick.color': '#000000',
            'font.family': 'serif', 'font.size': 11
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(port_df["timestamp"], port_df["balance"], color="#1f77b4", lw=2.5, label="Informed Follower Strategy")
        
        np.random.seed(42)
        random_pnl = np.random.choice([-self.trade_size, (1-0.5)/0.5 * self.trade_size], size=len(port_df))
        rand_balance = 10000.0 + np.cumsum(random_pnl)
        ax.plot(port_df["timestamp"], rand_balance, color="grey", lw=1.5, alpha=0.8, linestyle="--", label="Random Follower (50/50)")
        
        ax.set_title(f"Backtest Equity Curve\n(Start: $10,000 | Latency: {self.latency.total_seconds()/60:.0f}m)")
        ax.set_ylabel("Portfolio Value (USDC)")
        ax.legend(frameon=True, edgecolor='#000000', facecolor='#ffffff')
        ax.grid(True)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        fig.tight_layout()
        
        plot_path = PROJECT_ROOT / "results" / "plots" / "backtest_equity.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved Equity Curve plot to {plot_path.name}")


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    bt = StrategyBacktester(latency_minutes=5)
    bt.load_data()
    bt.simulate()
