"""
src/labeling/resolution_based.py
---------------------------------
P1.1: Label traders based on their actual prediction accuracy on resolved markets.

This replaces the circular composite-rank label (where features == label basis)
with a genuinely independent signal: did the trader's trades correctly anticipate
each market's final resolution?

Design:
  • For each trader, per resolved market:
      - BUY trade at price p, outcome = YES  →  correct prediction (+1)
      - BUY trade at price p, outcome = NO   →  incorrect (-1)
      - SELL trade at price p, outcome = NO  →  correct (+1)  [sold overpriced YES tokens]
      - SELL trade at price p, outcome = YES →  incorrect (-1) [sold the winning side]
  • informed_accuracy = sum(correct) / n_resolved_trades   ∈ [0, 1]
  • A trader is "Informed" if:
        informed_accuracy >= INFORMED_ACCURACY_THRESHOLD   (default 0.60)
        AND n_resolved_trades >= MIN_RESOLVED_TRADES       (default 3)
  • Eligible traders without enough resolved trades → label = -1 (excluded from training)
  • All other traders (zero resolved trades) → label = 0 (Noise)

Fallback:
  If the enriched file or outcome column is unavailable, falls back to composite rank.
"""

from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ── Resolution normalization ────────────────────────────────────────────────

_YES_TOKENS = {"yes", "up", "true", "1", "higher", "over", "win"}
_NO_TOKENS  = {"no", "down", "false", "0", "lower", "under", "lose"}


def _normalise_outcome(outcome_str: str) -> str:
    """Map raw outcome string to canonical 'YES' or 'NO'."""
    if not outcome_str or outcome_str in ("__OPEN__", "nan"):
        return "OPEN"
    s = str(outcome_str).strip().lower()
    if s in _YES_TOKENS:
        return "YES"
    if s in _NO_TOKENS:
        return "NO"
    # Numeric probability: >0.5 = YES resolved
    try:
        v = float(s)
        return "YES" if v > 0.5 else "NO"
    except ValueError:
        pass
    return "UNKNOWN"


# ── Core labeler ────────────────────────────────────────────────────────────

def compute_resolution_labels(
    trades_df: pd.DataFrame,
    resolutions: dict,
    min_resolved_trades: int = 3,
    accuracy_threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Compute per-trader prediction accuracy on resolved markets.

    Parameters
    ----------
    trades_df : DataFrame with columns [address, market_id, side, outcome, price, amount]
    resolutions : dict {market_id → raw_resolution_string}
    min_resolved_trades : minimum resolved-market trades for label eligibility
    accuracy_threshold : min accuracy to be labelled "Informed"

    Returns
    -------
    DataFrame with columns:
        address, n_resolved, n_correct, accuracy,
        Trader_Success_Rate (0/1/-1), label_source
    """
    logger.info(f"Computing resolution-based labels (min_trades={min_resolved_trades}, "
                f"threshold={accuracy_threshold:.0%}) …")

    # Build canonical resolution map
    res_map = {mid: _normalise_outcome(res) for mid, res in resolutions.items()}

    has_outcome_col = "outcome" in trades_df.columns

    records = []
    for address, grp in trades_df.groupby("address"):
        n_resolved = 0
        n_correct  = 0

        for _, row in grp.iterrows():
            market_res = res_map.get(row["market_id"], "OPEN")
            if market_res not in ("YES", "NO"):
                continue  # skip unresolved / unknown

            n_resolved += 1
            side = str(row["side"]).upper()

            # Determine prediction from trade side + outcome field (if available)
            if has_outcome_col and pd.notna(row.get("outcome")):
                trade_outcome = _normalise_outcome(str(row["outcome"]))
            else:
                # Fallback heuristic: BUY at price > 0.5 predicts YES, < 0.5 predicts NO
                price = float(row.get("price", 0.5))
                if side == "BUY":
                    trade_outcome = "YES" if price >= 0.5 else "NO"
                else:  # SELL — selling YES tokens means predicting NO
                    trade_outcome = "NO" if price >= 0.5 else "YES"

            if trade_outcome == market_res:
                n_correct += 1

        accuracy = n_correct / n_resolved if n_resolved > 0 else 0.0

        if n_resolved >= min_resolved_trades:
            label = 1 if accuracy >= accuracy_threshold else 0
            source = "resolution"
        elif n_resolved > 0:
            label = -1   # not enough data → exclude from training
            source = "resolution_insufficient"
        else:
            label = 0    # no resolved market activity → noise
            source = "no_resolved_data"

        records.append({
            "address": address,
            "n_resolved": n_resolved,
            "n_correct": n_correct,
            "accuracy": round(accuracy, 4),
            "Trader_Success_Rate": label,
            "label_source": source,
        })

    result = pd.DataFrame(records)
    eligible = result[result["Trader_Success_Rate"] != -1]
    informed = result[result["Trader_Success_Rate"] == 1]
    insuf    = result[result["Trader_Success_Rate"] == -1]

    logger.info(
        f"Label summary: {len(informed):,} Informed | "
        f"{len(eligible) - len(informed):,} Noise | "
        f"{len(insuf):,} excluded (insufficient data) | "
        f"Total: {len(result):,}"
    )
    if len(eligible) > 0:
        logger.info(f"Informed rate among eligible: {len(informed)/len(eligible):.1%}")

    return result


# ── Pipeline integration helper ─────────────────────────────────────────────

def load_resolutions() -> dict:
    """Load market resolution data from standard CSV path."""
    res_path = DATA_DIR / "processed" / "market_resolutions.csv"
    if not res_path.exists():
        logger.warning("market_resolutions.csv not found.")
        return {}
    df = pd.read_csv(res_path)
    return dict(zip(df["market_id"], df["resolution"]))


def apply_resolution_labels(
    features_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    min_resolved_trades: int = 3,
    accuracy_threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Merge resolution-based labels into features_df.

    Traders labelled -1 (insufficient data) are dropped from the output
    so models are only trained on addressess with ground-truth labels.
    """
    resolutions = load_resolutions()
    label_df = compute_resolution_labels(
        trades_df, resolutions,
        min_resolved_trades=min_resolved_trades,
        accuracy_threshold=accuracy_threshold,
    )

    merged = features_df.merge(
        label_df[["address", "n_resolved", "n_correct", "accuracy",
                  "Trader_Success_Rate", "label_source"]],
        on="address", how="left",
    )

    # -1 = not enough resolved trades → exclude
    before = len(merged)
    merged = merged[merged["Trader_Success_Rate"] != -1].copy()
    logger.info(f"Dropped {before - len(merged):,} traders with insufficient resolved trades.")

    return merged
