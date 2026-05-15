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
from dataclasses import asdict, dataclass
from numbers import Number
from typing import Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class BacktestAssumptions:
    """Explicit assumptions used by the copy-trading simulator."""

    initial_balance: float = 10000.0
    latency_minutes: int = 5
    trade_size: float = 100.0
    fees_pct: float = 0.001
    max_trade_fraction_of_balance: float = 0.02
    max_market_exposure_fraction: float = 0.10
    liquidity_lookback_minutes: int = 60
    max_participation_rate: float = 0.10
    min_fill_notional: float = 1.0
    price_impact_bps: float = 10.0
    settlement: str = "binary_payout_at_resolution"
    execution_price: str = "last_observed_trade_price_at_or_before_latency_timestamp"
    copied_side: str = "BUY_only"
    expert_bet_rule: str = "price>=0.5 implies YES else NO"
    liquidity_model: str = "capped_by_recent_market_volume_and_participation_rate"
    slippage_model: str = "adverse_price_impact_from_participation_rate"
    capital_model: str = "single_trade_and_market_exposure_caps"
    market_filter: str = "resolved_yes_no_only"


LEDGER_COLUMNS = [
    "timestamp",
    "market_id",
    "expert_address",
    "strategy",
    "exec_price",
    "expert_price",
    "expert_bet",
    "final_resolution",
    "requested_notional",
    "filled_notional",
    "recent_market_volume",
    "participation_rate",
    "liquidity_cap",
    "capital_cap",
    "market_exposure_before",
    "market_exposure_after",
    "slippage",
    "fee",
    "gross_profit",
    "net_profit",
    "balance",
]


def market_concentration_metrics(port_df: pd.DataFrame) -> Dict:
    """Summarize filled-notional concentration across markets."""
    if port_df.empty or "market_id" not in port_df.columns or "filled_notional" not in port_df.columns:
        return {
            "Unique_Markets": 0,
            "Top_Market_Filled_Notional_Share": 0.0,
            "Market_HHI": 0.0,
            "Effective_Markets": 0.0,
        }

    exposure = pd.to_numeric(port_df["filled_notional"], errors="coerce").fillna(0.0)
    market_exposure = exposure.groupby(port_df["market_id"]).sum()
    market_exposure = market_exposure[market_exposure > 0]
    total = float(market_exposure.sum())
    if total <= 0:
        return {
            "Unique_Markets": 0,
            "Top_Market_Filled_Notional_Share": 0.0,
            "Market_HHI": 0.0,
            "Effective_Markets": 0.0,
        }

    shares = market_exposure / total
    hhi = float((shares ** 2).sum())
    return {
        "Unique_Markets": int(len(shares)),
        "Top_Market_Filled_Notional_Share": float(shares.max()),
        "Market_HHI": hhi,
        "Effective_Markets": float(1.0 / hhi) if hhi > 0 else 0.0,
    }


class StrategyBacktester:
    def __init__(self, 
                 latency_minutes: int = 5,
                 trade_size: float = 100.0,
                 fees_pct: float = 0.001,
                 initial_balance: float = 10000.0,
                 max_trade_fraction_of_balance: float = 0.02,
                 max_market_exposure_fraction: float = 0.10,
                 liquidity_lookback_minutes: int = 60,
                 max_participation_rate: float = 0.10,
                 min_fill_notional: float = 1.0,
                 price_impact_bps: float = 10.0):
        self.latency = pd.Timedelta(minutes=latency_minutes)
        self.trade_size = trade_size
        self.fees = fees_pct
        self.assumptions = BacktestAssumptions(
            initial_balance=initial_balance,
            latency_minutes=latency_minutes,
            trade_size=trade_size,
            fees_pct=fees_pct,
            max_trade_fraction_of_balance=max_trade_fraction_of_balance,
            max_market_exposure_fraction=max_market_exposure_fraction,
            liquidity_lookback_minutes=liquidity_lookback_minutes,
            max_participation_rate=max_participation_rate,
            min_fill_notional=min_fill_notional,
            price_impact_bps=price_impact_bps,
        )
        
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
        self.tx_df["timestamp"] = pd.to_datetime(
            self.tx_df["timestamp"], errors="coerce", utc=True
        ).dt.tz_convert(None)
        self.tx_df = self.tx_df.dropna(subset=["timestamp"])
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

    def simulate(self, output_prefix: str = "backtest"):
        logger.info("Initializing Backtest simulation...")
        return self.simulate_for_addresses(
            target_addresses=self.experts,
            strategy_name="informed_copy",
            output_prefix=output_prefix,
        )

    def simulate_for_addresses(
        self,
        target_addresses: Iterable[str],
        strategy_name: str,
        tx_df: Optional[pd.DataFrame] = None,
        window_start=None,
        window_end=None,
        output_prefix: Optional[str] = None,
    ) -> Dict:
        """Simulate copying BUY trades from target addresses, optionally inside a time window."""
        source_tx = self.tx_df if tx_df is None else tx_df
        target_addresses = set(target_addresses)
        if not target_addresses:
            return self._write_no_trade_outputs(
                output_prefix=output_prefix,
                strategy_name=strategy_name,
                reason="No target addresses supplied.",
            )

        working_tx = source_tx.copy()
        working_tx["timestamp"] = pd.to_datetime(
            working_tx["timestamp"], errors="coerce", utc=True
        ).dt.tz_convert(None)
        working_tx = working_tx.dropna(subset=["timestamp"])
        if window_start is not None:
            working_tx = working_tx[working_tx["timestamp"] >= pd.Timestamp(window_start)]
        if window_end is not None:
            working_tx = working_tx[working_tx["timestamp"] < pd.Timestamp(window_end)]

        if "volume" not in working_tx.columns:
            working_tx["volume"] = working_tx["amount"] * working_tx["price"]
        working_tx["volume"] = pd.to_numeric(working_tx["volume"], errors="coerce").fillna(0.0)
        working_tx["amount"] = pd.to_numeric(working_tx["amount"], errors="coerce").fillna(0.0)
        working_tx["price"] = pd.to_numeric(working_tx["price"], errors="coerce")
        working_tx = working_tx.dropna(subset=["price"])

        expert_tx = working_tx[working_tx["address"].isin(target_addresses)].copy()
        expert_tx = expert_tx.sort_values("timestamp")
        
        target_markets = set(expert_tx["market_id"])
        
        # Build price interpolator for target markets
        market_series = {
            m: grp.set_index("timestamp")[["price", "side"]]
            for m, grp in working_tx[working_tx["market_id"].isin(target_markets)].groupby("market_id")
        }

        portfolio = []
        
        balance = self.assumptions.initial_balance
        peak_balance = balance
        max_drawdown = 0.0
        skipped_liquidity = 0
        skipped_capital = 0
        requested_trades = 0
        capital_capped_trades = 0
        liquidity_capped_trades = 0
        market_exposure_capped_trades = 0
        market_exposure: Dict[str, float] = {}
        
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
            requested_trades += 1
            requested_notional = float(self.trade_size)
            capital_cap = max(0.0, balance * self.assumptions.max_trade_fraction_of_balance)
            market_cap = max(0.0, balance * self.assumptions.max_market_exposure_fraction)
            market_exposure_before = float(market_exposure.get(m_id, 0.0))
            remaining_market_cap = max(0.0, market_cap - market_exposure_before)

            liquidity_start = t_exec - pd.Timedelta(minutes=self.assumptions.liquidity_lookback_minutes)
            recent_market = working_tx[
                (working_tx["market_id"] == m_id)
                & (working_tx["timestamp"] >= liquidity_start)
                & (working_tx["timestamp"] <= t_exec)
            ]
            recent_market_volume = float(recent_market["volume"].sum())
            liquidity_cap = recent_market_volume * self.assumptions.max_participation_rate
            fill_notional = min(requested_notional, capital_cap, remaining_market_cap, liquidity_cap)

            if fill_notional < requested_notional:
                if fill_notional == capital_cap:
                    capital_capped_trades += 1
                if fill_notional == remaining_market_cap:
                    market_exposure_capped_trades += 1
                if fill_notional == liquidity_cap:
                    liquidity_capped_trades += 1

            if fill_notional < self.assumptions.min_fill_notional:
                if liquidity_cap < self.assumptions.min_fill_notional:
                    skipped_liquidity += 1
                else:
                    skipped_capital += 1
                continue

            participation_rate = fill_notional / recent_market_volume if recent_market_volume > 0 else 0.0
            adverse_impact = (self.assumptions.price_impact_bps / 10000.0) * (1.0 + participation_rate)
            our_price = min(0.99, max(0.01, float(our_price) + adverse_impact))
            
            if expert_bet == final_res:
                profit = (1.0 - our_price) * (fill_notional / our_price)
            else:
                profit = -fill_notional
                
            fee = fill_notional * self.fees
            net_profit = profit - fee
            
            balance += net_profit
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd
            market_exposure_after = market_exposure_before + fill_notional
            market_exposure[m_id] = market_exposure_after
                
            portfolio.append({
                "timestamp": t_exec,
                "market_id": m_id,
                "expert_address": trade["address"],
                "strategy": strategy_name,
                "exec_price": our_price,
                "expert_price": expert_price,
                "expert_bet": expert_bet,
                "final_resolution": final_res,
                "requested_notional": requested_notional,
                "filled_notional": fill_notional,
                "recent_market_volume": recent_market_volume,
                "participation_rate": participation_rate,
                "liquidity_cap": liquidity_cap,
                "capital_cap": capital_cap,
                "market_exposure_before": market_exposure_before,
                "market_exposure_after": market_exposure_after,
                "slippage": our_price - expert_price if expert_bet == "YES" else expert_price - our_price,
                "fee": fee,
                "gross_profit": profit,
                "net_profit": net_profit,
                "balance": balance
            })

        if not portfolio:
            return self._write_no_trade_outputs(
                output_prefix=output_prefix,
                strategy_name=strategy_name,
                reason=(
                    "No BUY trades from target addresses matched resolved YES/NO markets "
                    "after capital and liquidity constraints."
                ),
                diagnostics={
                    "Requested_Trades": requested_trades,
                    "Total_Requested_Notional": requested_trades * self.trade_size,
                    "Skipped_Liquidity": skipped_liquidity,
                    "Skipped_Capital": skipped_capital,
                    "Capital_Capped_Trades": capital_capped_trades,
                    "Liquidity_Capped_Trades": liquidity_capped_trades,
                    "Market_Exposure_Capped_Trades": market_exposure_capped_trades,
                },
            )

        port_df = pd.DataFrame(portfolio).sort_values("timestamp")
        
        metrics = {
            "status": "ok",
            "Strategy": strategy_name,
            "Total_Trades": len(port_df),
            "Requested_Trades": requested_trades,
            "Win_Rate": (port_df["net_profit"] > 0).mean(),
            "Total_Net_Profit": port_df["net_profit"].sum(),
            "Total_Requested_Notional": requested_trades * self.trade_size,
            "Total_Filled_Notional": port_df["filled_notional"].sum(),
            "Fill_Rate": port_df["filled_notional"].sum() / ((requested_trades * self.trade_size) + 1e-5),
            "Skipped_Liquidity": skipped_liquidity,
            "Skipped_Capital": skipped_capital,
            "Capital_Capped_Trades": capital_capped_trades,
            "Liquidity_Capped_Trades": liquidity_capped_trades,
            "Market_Exposure_Capped_Trades": market_exposure_capped_trades,
            "ROI_Pct": port_df["net_profit"].sum() / (port_df["filled_notional"].sum() + 1e-5),
            "Max_Drawdown": max_drawdown,
            "Average_Slippage": port_df["slippage"].mean(),
            "assumptions": asdict(self.assumptions),
        }
        metrics.update(market_concentration_metrics(port_df))
        
        port_df["date"] = port_df["timestamp"].dt.date
        daily_pnl = port_df.groupby("date")["net_profit"].sum()
        if len(daily_pnl) > 1 and daily_pnl.std() > 0:
            metrics["Sharpe_Ratio (Annualized)"] = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(365)
        else:
            metrics["Sharpe_Ratio (Annualized)"] = 0.0

        logger.info("\n--- BACKTEST RESULTS ---")
        for k, v in metrics.items():
            if isinstance(v, Number) and ("Pct" in k or "Rate" in k):
                logger.info(f"{k}: {v:.2%}")
            elif isinstance(v, Number):
                logger.info(f"{k}: {v:.4f}")
            else:
                logger.info(f"{k}: {v}")
                
        if output_prefix:
            out_dir = PROJECT_ROOT / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            port_df.to_csv(out_dir / f"{output_prefix}_trades.csv", index=False)
            with open(out_dir / f"{output_prefix}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4, default=str)
            if output_prefix == "backtest":
                # Backward-compatible artifact names.
                port_df.to_csv(out_dir / "backtest_trades.csv", index=False)
                with open(out_dir / "backtest_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=4, default=str)
                self._plot_equity(port_df)
        return metrics

    def _write_no_trade_outputs(
        self,
        output_prefix: Optional[str],
        strategy_name: str,
        reason: str,
        diagnostics: Optional[Dict] = None,
    ) -> Dict:
        logger.warning(reason)
        diagnostics = diagnostics or {}
        metrics = {
            "status": "no_valid_trades",
            "Strategy": strategy_name,
            "Total_Trades": 0,
            "Requested_Trades": diagnostics.get("Requested_Trades", 0),
            "Win_Rate": 0.0,
            "Total_Net_Profit": 0.0,
            "Total_Requested_Notional": diagnostics.get("Total_Requested_Notional", 0.0),
            "Total_Filled_Notional": 0.0,
            "Fill_Rate": 0.0,
            "Skipped_Liquidity": diagnostics.get("Skipped_Liquidity", 0),
            "Skipped_Capital": diagnostics.get("Skipped_Capital", 0),
            "Capital_Capped_Trades": diagnostics.get("Capital_Capped_Trades", 0),
            "Liquidity_Capped_Trades": diagnostics.get("Liquidity_Capped_Trades", 0),
            "Market_Exposure_Capped_Trades": diagnostics.get("Market_Exposure_Capped_Trades", 0),
            "ROI_Pct": 0.0,
            "Max_Drawdown": 0.0,
            "Average_Slippage": 0.0,
            "Unique_Markets": 0,
            "Top_Market_Filled_Notional_Share": 0.0,
            "Market_HHI": 0.0,
            "Effective_Markets": 0.0,
            "Sharpe_Ratio (Annualized)": 0.0,
            "reason": reason,
            "assumptions": asdict(self.assumptions),
        }
        if output_prefix:
            out_dir = PROJECT_ROOT / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            empty = pd.DataFrame(columns=LEDGER_COLUMNS)
            empty.to_csv(out_dir / f"{output_prefix}_trades.csv", index=False)
            with open(out_dir / f"{output_prefix}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4, default=str)
            if output_prefix == "backtest":
                empty.to_csv(out_dir / "backtest_trades.csv", index=False)
                with open(out_dir / "backtest_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=4, default=str)
        return metrics

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
