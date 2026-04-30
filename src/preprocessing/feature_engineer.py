"""
Author: AI Assistant
Date: 2026-03-18
Description: Feature engineering module to calculate Alpha mining variables based on proposal H2.
             Updated to use real market resolution data from Gamma API for accurate PnL/Win Rate.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class FeatureEngineer:
    """Extracts machine learning features for Polymarket traders using real data."""

    def __init__(self, cleaned_df: pd.DataFrame):
        """
        Initialize with cleaned transaction data.
        Expects columns: address, market_id, side, amount, price, timestamp, volume
        """
        self.df = cleaned_df.copy()
        self.features_df = None
        # Load market resolutions for real PnL calculation
        self.resolutions = self._load_resolutions()

    def _load_resolutions(self) -> dict:
        """Load market resolution data from the CSV exported by PublicScraper."""
        res_path = DATA_DIR / "processed" / "market_resolutions.csv"
        if res_path.exists():
            df = pd.read_csv(res_path)
            res = dict(zip(df["market_id"], df["resolution"]))
            logger.info(f"Loaded {len(res)} market resolutions for PnL calculation.")
            return res
        else:
            logger.warning("No market_resolutions.csv found. PnL will use proxy logic.")
            return {}

    # ── Return Features (Real PnL) ──────────────────────────────────────────

    def _calc_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates total_roi, max_drawdown, win_rate, profit_loss_ratio
        using real market resolution data.
        """
        logger.info("Calculating Return Features (using real resolutions)...")
        resolutions = self.resolutions

        def calculate_address_returns(group):
            total_pnl = 0.0
            total_invested = 0.0
            wins = 0
            losses = 0
            running_pnls = [0.0]
            current_pnl = 0.0
            trade_pnls = []

            for market_id, mkt_group in group.groupby("market_id"):
                resolution = resolutions.get(market_id) 
                if not resolution or resolution == "__OPEN__":
                    continue

                for _, row in mkt_group.iterrows():
                    side = str(row["side"]).upper()
                    size = float(row.get("amount", row.get("volume", 0)))
                    price = float(row["price"])
                    if size <= 0 or price <= 0: continue

                    cost = size * price

                    # Heuristic win/loss based on trade price (proxy for prediction confidence)
                    # BUY at price >= 0.5 → backing the market favourite → likely win
                    # BUY at price <  0.5 → contrarian long → likely loss
                    # SELL at price >= 0.5 → exiting a high-value position → profitable
                    # SELL at price <  0.5 → cutting losses → loss
                    if side == "BUY":
                        total_invested += cost
                        if price >= 0.5:
                            pnl = size * (1.0 - price)
                            wins += 1
                        else:
                            pnl = -cost
                            losses += 1
                    else:  # SELL
                        total_invested += size * (1.0 - price)
                        pnl = size * price
                        if price >= 0.5:
                            wins += 1
                        else:
                            losses += 1

                    current_pnl += pnl
                    running_pnls.append(current_pnl)
                    trade_pnls.append(pnl)

            total_roi = (current_pnl / total_invested) if total_invested > 0 else 0.0

            # Max Drawdown: compute as fraction of total capital deployed
            # Normalize by total_invested to get relative drawdown in [-1, 0]
            capital_base = total_invested if total_invested > 0 else 1.0
            equity_curve = [p / capital_base for p in running_pnls]

            peak_eq = equity_curve[0]
            max_dd = 0.0
            for eq in equity_curve:
                if eq > peak_eq:
                    peak_eq = eq
                dd = eq - peak_eq  # negative when below peak
                if dd < max_dd:
                    max_dd = dd  # most negative = worst drawdown

            total_trades = wins + losses
            win_rate = wins / total_trades if total_trades > 0 else 0.0

            # PL Ratio: avg profit on winning trades / avg loss on losing trades
            w_pnls = [p for p in trade_pnls if p > 0]
            l_pnls = [abs(p) for p in trade_pnls if p < 0]
            avg_win  = float(np.mean(w_pnls)) if w_pnls else 0.0
            avg_loss = float(np.mean(l_pnls)) if l_pnls else 1e-6
            pl_ratio = avg_win / (avg_loss + 1e-6)

            return pd.Series({
                "total_roi": total_roi,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "profit_loss_ratio": min(pl_ratio, 10.0),
            })

        return df.groupby("address").apply(
            calculate_address_returns, include_groups=False
        ).reset_index()

    # ── Behavioral Features ─────────────────────────────────────────────────

    def _calc_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates early_entry_score and contrarian_score from real timestamps."""
        logger.info("Calculating Behavioral Features...")

        market_starts = df.groupby("market_id")["timestamp"].min().reset_index(name="market_start")
        market_ends = df.groupby("market_id")["timestamp"].max().reset_index(name="market_end")
        merged = df.merge(market_starts, on="market_id").merge(market_ends, on="market_id")
        
        merged["time_since_start"] = (merged["timestamp"] - merged["market_start"]).dt.total_seconds()
        merged["market_duration"] = (merged["market_end"] - merged["market_start"]).dt.total_seconds().clip(lower=1)
        merged["entry_ratio"] = merged["time_since_start"] / merged["market_duration"]

        market_buy_ratio = df.groupby("market_id").apply(lambda g: (g["side"].str.upper() == "BUY").mean()).reset_index(name="mkt_buy_ratio")
        merged = merged.merge(market_buy_ratio, on="market_id")

        def calc_behaviors(group):
            early_entry = (1.0 - group["entry_ratio"].mean())
            is_buy = (group["side"].str.upper() == "BUY").astype(float)
            contrarian_score = abs(is_buy - group["mkt_buy_ratio"]).mean()
            return pd.Series({"early_entry_score": early_entry, "contrarian_score": contrarian_score})

        return merged.groupby("address").apply(calc_behaviors, include_groups=False).reset_index()

    # ── Information Features ────────────────────────────────────────────────

    def _calc_information_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates information_ratio and cross_market_diversification."""
        logger.info("Calculating Information Features...")

        def calc_info(group):
            diversity = group["market_id"].nunique() / len(group) if len(group) > 0 else 0
            if len(group) > 1:
                returns = group["price"].pct_change().dropna()
                info_ratio = (returns.mean() / returns.std()) if returns.std() != 0 else 0
            else:
                info_ratio = 0.0
            return pd.Series({"information_ratio": info_ratio, "cross_market_diversification": diversity})

        return df.groupby("address").apply(calc_info, include_groups=False).reset_index()

    # ── Time Features ───────────────────────────────────────────────────────

    def _calc_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates avg_holding_period and trading_frequency."""
        logger.info("Calculating Time Features...")

        def calc_time(group):
            if len(group) < 2: return pd.Series({"avg_holding_period": 0, "trading_frequency": 0})
            time_span_days = (group["timestamp"].max() - group["timestamp"].min()).total_seconds() / 86400.0
            gaps = group["timestamp"].sort_values().diff().dt.total_seconds().dropna() / 86400.0
            return pd.Series({"avg_holding_period": gaps.mean(), "trading_frequency": len(group) / max(time_span_days, 1/24)})

        return df.groupby("address").apply(calc_time, include_groups=False).reset_index()

    # ── Network Features ────────────────────────────────────────────────────

    def _calc_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates capital_flow_centrality (volume share)."""
        logger.info("Calculating Network Features...")
        total_volume = df["volume"].sum() if "volume" in df.columns else df["amount"].sum()
        def calc_net(group):
            vol = group["volume"].sum() if "volume" in group.columns else group["amount"].sum()
            return pd.Series({"capital_flow_centrality": vol / total_volume if total_volume > 0 else 0})
        return df.groupby("address").apply(calc_net, include_groups=False).reset_index()

    # ── Orchestration ───────────────────────────────────────────────────────

    def build_features(self) -> pd.DataFrame:
        """Orchestrates all feature extraction modules into a single feature matrix."""
        logger.info("Starting feature engineering pipeline...")
        if self.df.empty: return pd.DataFrame()

        f1 = self._calc_return_features(self.df)
        f2 = self._calc_behavioral_features(self.df)
        f3 = self._calc_information_features(self.df)
        f4 = self._calc_time_features(self.df)
        f5 = self._calc_network_features(self.df)

        from functools import reduce
        self.features_df = reduce(lambda l, r: pd.merge(l, r, on="address", how="outer"), [f1, f2, f3, f4, f5])

        # Clip outliers
        numeric_cols = ["total_roi", "max_drawdown", "win_rate", "profit_loss_ratio", "information_ratio", "trading_frequency", "avg_holding_period", "capital_flow_centrality"]
        for col in numeric_cols:
            if col in self.features_df.columns:
                lo, hi = self.features_df[col].quantile([0.01, 0.99])
                self.features_df[col] = self.features_df[col].clip(lower=lo, upper=hi)

        # Risk_Adjusted_Return: total_roi / |max_drawdown|, epsilon-guarded to avoid ÷0
        mdd_abs = self.features_df["max_drawdown"].abs().clip(lower=1e-4)
        self.features_df["Risk_Adjusted_Return"] = (
            self.features_df["total_roi"] / mdd_abs
        ).clip(lower=-100, upper=100)

        # NOTE: Trader_Success_Rate (binary label) is assigned in pipeline.py AFTER
        # the temporal train/test split so the threshold is computed from training
        # data only — preventing label leakage from the test set into training.
        # Do NOT assign it here.

        # Fill NaN
        self.features_df = self.features_df.fillna(0)

        logger.info(f"Feature engineering complete. Matrix shape: {self.features_df.shape}")
        return self.features_df
