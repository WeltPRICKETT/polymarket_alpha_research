"""
Author: AI Assistant
Date: 2026-03-17 (refactored 2026-04-11)
Description: Pipeline script — data cleaning → feature engineering → labeling → export.

Improvements (2026-04-11):
  P0.2  Uses centralised settings for cleaner params.
  P1.1  Primary label: resolution-based prediction accuracy on resolved markets.
        Fallback label: composite rank (roi + win_rate + plr) when resolved data insufficient.
  P2.1  Label is never computed from full dataset; always train-set-based where applicable.
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config.settings import (
    LOG_LEVEL, TRAIN_RATIO, VAL_RATIO,
    CLEANER_MIN_TRADES, CLEANER_MIN_VOLUME, CLEANER_TOP_PERCENTILE,
    MIN_RESOLVED_TRADES, INFORMED_ACCURACY_THRESHOLD, LABEL_TOP_PERCENTILE,
    FUTURE_LABEL_OBSERVATION_DAYS, FUTURE_LABEL_HORIZON_DAYS,
    FUTURE_LABEL_MIN_RESOLVED_TRADES, FUTURE_LABEL_TOP_PERCENTILE,
)
from src.data_ingestion.storage import Storage
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.feature_engineer import FeatureEngineer
from src.labeling.resolution_based import apply_resolution_labels
from src.labeling.future_return import (
    FutureReturnLabelConfig,
    apply_future_return_labels,
    assign_future_return_threshold,
    build_observation_transactions,
)
from src.labeling.audit import write_label_audit
from src.data_quality.audit import write_audit_artifacts
from src.data_ingestion.external_sources import build_wallet_event_features

logger.remove()
logger.add(sys.stdout, level=LOG_LEVEL)


# ── Temporal split ────────────────────────────────────────────────────────────

def temporal_train_test_split(
    cleaned_df: pd.DataFrame,
    features_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> pd.DataFrame:
    """
    Strict temporal train/val/test split on first-trade-date of each trader.

    - Train: traders whose first trade is before the training cutoff.
    - Val:   traders between training cutoff and validation cutoff.
    - Test:  traders after translation cutoff.

    Prevents any future-data leakage into model features or labels.
    """
    logger.info("Performing strict temporal train/val/test split …")

    first_trade = (
        cleaned_df.groupby("address")["timestamp"]
        .min()
        .reset_index(name="first_trade_date")
    )

    global_min = cleaned_df["timestamp"].min()
    global_max = cleaned_df["timestamp"].max()
    time_span = global_max - global_min

    train_cutoff = global_min + time_span * train_ratio
    val_cutoff = global_min + time_span * (train_ratio + val_ratio)

    logger.info(f"Time range: {global_min} → {global_max}")
    logger.info(f"Train Cutoff: {train_cutoff} ({train_ratio*100:.0f}%)")
    logger.info(f"Val Cutoff:   {val_cutoff} ({(train_ratio+val_ratio)*100:.0f}%)")

    df = features_df.merge(first_trade, on="address", how="left")
    
    # Categorize splits
    df["split"] = "test"
    df.loc[df["first_trade_date"] < train_cutoff, "split"] = "train"
    df.loc[(df["first_trade_date"] >= train_cutoff) & (df["first_trade_date"] < val_cutoff), "split"] = "val"

    n_train = (df["split"] == "train").sum()
    n_val   = (df["split"] == "val").sum()
    n_test  = (df["split"] == "test").sum()
    logger.info(f"Split → Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")
    
    # Backward compat: keep is_train for existing logic that expects it
    df["is_train"] = df["split"] == "train"
    
    return df


def temporal_resplit_labeled_features(
    labeled_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> pd.DataFrame:
    """
    Recompute train/val/test on the labeled sample itself.

    Future-return labels drop right-censored rows. Splitting before that drop can
    erase val/test entirely, so this is used after label eligibility is known.
    """
    df = labeled_df.copy()
    df["first_trade_date"] = pd.to_datetime(df["first_trade_date"], errors="coerce")
    df = df.dropna(subset=["first_trade_date"])
    if df.empty:
        return df

    global_min = df["first_trade_date"].min()
    global_max = df["first_trade_date"].max()
    time_span = global_max - global_min
    train_cutoff = global_min + time_span * train_ratio
    val_cutoff = global_min + time_span * (train_ratio + val_ratio)

    df["split"] = "test"
    df.loc[df["first_trade_date"] < train_cutoff, "split"] = "train"
    df.loc[(df["first_trade_date"] >= train_cutoff) & (df["first_trade_date"] < val_cutoff), "split"] = "val"
    df["is_train"] = df["split"] == "train"
    logger.info(
        "Future-label eligible temporal split → Train: {:,} | Val: {:,} | Test: {:,}".format(
            int((df["split"] == "train").sum()),
            int((df["split"] == "val").sum()),
            int((df["split"] == "test").sum()),
        )
    )
    return df


# ── Fallback composite label ──────────────────────────────────────────────────

def _composite_label_for_split(subset: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Composite rank label: top LABEL_TOP_PERCENTILE by mean of three rank-percentiles.
    Traders with no resolved-market PnL activity get label=0.
    Applied independently per split to give both train/test ~20% positives.
    """
    active_mask = (
        (subset["total_roi"] != 0)
        | (subset["win_rate"] != 0)
        | (subset["profit_loss_ratio"] != 0)
    )
    active   = subset[active_mask].copy()
    inactive = subset[~active_mask].copy()

    inactive["Trader_Success_Rate"] = 0
    inactive["composite_score"]     = 0.0
    inactive["label_source"]        = "composite_inactive"

    if len(active) == 0:
        return pd.concat([active, inactive]).sort_index()

    composite = (
        active["total_roi"].rank(pct=True)
        + active["win_rate"].rank(pct=True)
        + active["profit_loss_ratio"].rank(pct=True)
    ) / 3.0

    threshold = composite.quantile(1 - LABEL_TOP_PERCENTILE)
    active["Trader_Success_Rate"] = (composite >= threshold).astype(int)
    active["composite_score"]     = composite.values
    active["label_source"]        = "composite_rank"

    n_pos = active["Trader_Success_Rate"].sum()
    logger.info(
        f"[Composite/{split_name}] active={len(active):,}/{len(subset):,} | "
        f"Informed={n_pos:,}/{len(active):,} ({n_pos/len(active):.1%})"
    )
    return pd.concat([active, inactive]).sort_index()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main(top_percentile: float = None, label_mode: str = "auto"):
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    top_percentile : float or None
        If set, keeps only the top N% of traders by volume.
    label_mode : str
        'resolution' — use P1.1 resolution-based labels (requires ≥MIN_RESOLVED_TRADES).
        'future_return' — use independent future-window return labels.
        'composite'  — use composite-rank labels (v2 fallback).
        'auto'       — try resolution first; fall back to composite if data insufficient.
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Preprocessing & Feature Engineering")
    logger.info(f"Label mode: {label_mode} | Train ratio: {TRAIN_RATIO}")
    logger.info("=" * 60)

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # ── Step 1: Load raw transactions ────────────────────────────────────────
    storage = Storage()
    raw_df = storage.load_transactions_df()
    logger.info(f"Loaded {len(raw_df):,} transactions from DB.")

    if raw_df.empty:
        logger.error("No transactions in DB. Run data collection first.")
        return

    # ── Step 2: Clean ────────────────────────────────────────────────────────
    cleaner = DataCleaner(raw_df)
    cleaned_df = cleaner.clean(
        min_trades=CLEANER_MIN_TRADES,
        min_volume=CLEANER_MIN_VOLUME,
        top_percentile=top_percentile or CLEANER_TOP_PERCENTILE,
    )

    clean_suffix = f"_top{int(top_percentile*100)}" if top_percentile else ""
    clean_path = BASE_DIR / "data" / "processed" / f"cleaned_transactions{clean_suffix}.csv"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(clean_path, index=False)
    logger.info(f"Cleaned data saved → {clean_path.name}  ({len(cleaned_df):,} rows)")

    # ── Step 3: Feature engineering ──────────────────────────────────────────
    feature_source_df = cleaned_df
    if label_mode == "future_return":
        feature_source_df = build_observation_transactions(
            cleaned_df,
            observation_days=FUTURE_LABEL_OBSERVATION_DAYS,
        )
        logger.info(
            "Future-return mode: feature source limited to first "
            f"{FUTURE_LABEL_OBSERVATION_DAYS} day(s) per wallet "
            f"({len(feature_source_df):,}/{len(cleaned_df):,} trades)."
        )

    engineer = FeatureEngineer(feature_source_df)
    features_df = engineer.build_features()
    market_event_map_path = BASE_DIR / "data" / "processed" / "sii_market_event_map.csv"
    if market_event_map_path.exists():
        try:
            market_event_map = pd.read_csv(market_event_map_path, low_memory=False)
            event_features = build_wallet_event_features(feature_source_df, market_event_map)
            features_df = features_df.merge(event_features, on="address", how="left")
            event_cols = [col for col in event_features.columns if col != "address"]
            features_df[event_cols] = features_df[event_cols].fillna(0)
            logger.info(
                "Merged SII event-context wallet features "
                f"({len(event_cols)} columns from {market_event_map_path.name})."
            )
        except Exception as e:
            logger.warning(f"Failed to merge SII event-context wallet features: {e}")

    if features_df.empty:
        logger.error("Feature matrix is empty. Aborting.")
        return

    logger.info("Feature matrix:\n" + str(features_df.describe().T.round(4)))

    # ── Step 4: Temporal split ───────────────────────────────────────────────
    df = temporal_train_test_split(cleaned_df, features_df, train_ratio=TRAIN_RATIO)

    # ── Step 5: Label assignment ─────────────────────────────────────────────
    use_resolution = False
    label_strategy = "composite"

    if label_mode == "future_return":
        df = apply_future_return_labels(
            df,
            cleaned_df,
            observation_days=FUTURE_LABEL_OBSERVATION_DAYS,
            horizon_days=FUTURE_LABEL_HORIZON_DAYS,
            min_future_resolved_trades=FUTURE_LABEL_MIN_RESOLVED_TRADES,
            top_percentile=FUTURE_LABEL_TOP_PERCENTILE,
        )
        df = temporal_resplit_labeled_features(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)
        df = assign_future_return_threshold(
            df,
            top_percentile=FUTURE_LABEL_TOP_PERCENTILE,
            config=FutureReturnLabelConfig(
                observation_days=FUTURE_LABEL_OBSERVATION_DAYS,
                horizon_days=FUTURE_LABEL_HORIZON_DAYS,
                min_future_resolved_trades=FUTURE_LABEL_MIN_RESOLVED_TRADES,
                top_percentile=FUTURE_LABEL_TOP_PERCENTILE,
            ),
        )
        label_strategy = "future_return"

    if label_mode in ("resolution", "auto"):
        # Attempt resolution-based labels (P1.1)
        labeled = apply_resolution_labels(
            df, cleaned_df,
            min_resolved_trades=MIN_RESOLVED_TRADES,
            accuracy_threshold=INFORMED_ACCURACY_THRESHOLD,
        )

        n_resolution_labeled = (labeled["label_source"] == "resolution").sum()
        resolution_rate = n_resolution_labeled / len(labeled) if len(labeled) > 0 else 0

        if resolution_rate >= 0.10:   # ≥10% traders have enough resolved data
            logger.info(
                f"✅ Resolution labels applied to {n_resolution_labeled:,} traders "
                f"({resolution_rate:.1%}).  Using P1.1 label mode."
            )
            df = labeled
            use_resolution = True
            label_strategy = "resolution"
        elif label_mode == "resolution":
            logger.warning(
                f"Resolution labels only cover {resolution_rate:.1%} of traders. "
                f"Consider running with label_mode='composite' until more markets resolve."
            )
            df = labeled
            use_resolution = True
            label_strategy = "resolution"
        else:
            logger.info(
                f"Resolution coverage too low ({resolution_rate:.1%}). "
                f"Falling back to composite-rank labels."
            )

    if label_strategy == "composite" and not use_resolution:
        # Composite-rank labels — independent per split
        train_idx = df["split"] == "train"
        val_idx   = df["split"] == "val"
        test_idx  = df["split"] == "test"
        
        train_labeled = _composite_label_for_split(df.loc[train_idx].copy(), "TRAIN")
        val_labeled   = _composite_label_for_split(df.loc[val_idx].copy(),   "VAL")
        test_labeled  = _composite_label_for_split(df.loc[test_idx].copy(),  "TEST")
        
        df = pd.concat([train_labeled, val_labeled, test_labeled]).sort_index()

    # ── Step 6: Final stats & export ─────────────────────────────────────────
    df["Trader_Success_Rate"] = df["Trader_Success_Rate"].fillna(0).astype(int)

    train_mask = df["split"] == "train"
    val_mask   = df["split"] == "val"
    test_mask  = df["split"] == "test"
    n_train    = train_mask.sum()
    n_val      = val_mask.sum()
    n_test     = test_mask.sum()
    train_pos  = df.loc[train_mask, "Trader_Success_Rate"].sum()
    val_pos    = df.loc[val_mask,   "Trader_Success_Rate"].sum()
    test_pos   = df.loc[test_mask,  "Trader_Success_Rate"].sum()

    suffix = f"_top{int(top_percentile*100)}" if top_percentile else ""
    out_path = BASE_DIR / "data" / "features" / f"model_input{suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    write_label_audit(df, label_strategy=label_strategy, requested_label_mode=label_mode)

    logger.info("=" * 60)
    logger.info(f"✅  Phase 2 complete → {out_path.name}")
    logger.info(f"    Traders:  {len(df):,}  (train={n_train:,}  val={n_val:,}  test={n_test:,})")
    logger.info(f"    Labels:   train_pos={train_pos:,}/{n_train:,} ({train_pos/(n_train or 1):.1%})"
                f"  |  val_pos={val_pos:,}/{n_val:,} ({val_pos/(n_val or 1):.1%})"
                f"  |  test_pos={test_pos:,}/{n_test:,} ({test_pos/(n_test or 1):.1%})")
    if label_strategy == "future_return":
        strategy_label = "future-return independent"
    elif use_resolution:
        strategy_label = "resolution-based (P1.1)"
    else:
        strategy_label = "composite-rank (fallback)"
    logger.info(f"    Label strategy: {strategy_label}")
    logger.info(f"    NaNs in matrix: {df.isna().sum().sum()}")
    logger.info("=" * 60)

    try:
        write_audit_artifacts()
    except Exception as e:
        logger.warning(f"Data quality audit failed after preprocessing: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing & Feature Engineering Pipeline")
    parser.add_argument("--top-percentile", type=float, default=None,
                        help="Keep only top N%% traders by volume (e.g. 0.1 for top 10%%)")
    parser.add_argument("--label-mode", choices=["auto", "resolution", "future_return", "composite"],
                        default="auto",
                        help="Label strategy: auto (default), resolution, future_return, or composite")
    args = parser.parse_args()
    main(top_percentile=args.top_percentile, label_mode=args.label_mode)
