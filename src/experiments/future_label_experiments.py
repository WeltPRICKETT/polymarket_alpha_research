"""
Author: AI Assistant
Date: 2026-05-13
Description: Phase 8-12 future-label sensitivity, rolling-origin, and model-informed backtests.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.backtesting.equal_capital import EQUAL_CAPITAL_METHOD, add_equal_capital_metrics
from src.backtesting.engine import StrategyBacktester
from src.labeling.future_return import (
    FutureReturnLabelConfig,
    assign_future_return_threshold,
    compute_future_return_labels,
)
from src.labeling.resolution_based import load_resolutions
from src.models.feature_sets import get_feature_columns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalise_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)


def temporal_resplit(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> pd.DataFrame:
    """Assign temporal train/val/test splits to an eligible label frame."""
    out = df.copy()
    out["first_trade_date"] = _normalise_time(out["first_trade_date"])
    out = out.dropna(subset=["first_trade_date"])
    if out.empty:
        out["split"] = []
        return out

    start = out["first_trade_date"].min()
    end = out["first_trade_date"].max()
    span = end - start
    train_cutoff = start + span * train_ratio
    val_cutoff = start + span * (train_ratio + val_ratio)
    out["split"] = "test"
    out.loc[out["first_trade_date"] < train_cutoff, "split"] = "train"
    out.loc[
        (out["first_trade_date"] >= train_cutoff) & (out["first_trade_date"] < val_cutoff),
        "split",
    ] = "val"
    out["is_train"] = out["split"] == "train"
    return out


def summarise_labeled_frame(df: pd.DataFrame) -> Dict:
    """Return split health and label distribution for a labeled frame."""
    summary: Dict = {
        "rows": int(len(df)),
        "split_counts": {},
        "positive_counts": {},
        "single_class_splits": [],
        "test_rows": 0,
        "test_positive": 0,
    }
    if df.empty or "split" not in df.columns:
        return summary

    for split, subset in df.groupby("split"):
        labels = subset["Trader_Success_Rate"]
        summary["split_counts"][str(split)] = int(len(subset))
        summary["positive_counts"][str(split)] = int((labels == 1).sum())
        if labels.nunique(dropna=False) < 2:
            summary["single_class_splits"].append(str(split))
    summary["test_rows"] = summary["split_counts"].get("test", 0)
    summary["test_positive"] = summary["positive_counts"].get("test", 0)
    return summary


def build_future_label_frame(
    trades_df: pd.DataFrame,
    resolutions: Dict[str, str],
    config: FutureReturnLabelConfig,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> pd.DataFrame:
    """Build eligible future-return labels, temporal split, and train-threshold labels."""
    raw = compute_future_return_labels(trades_df, resolutions, config)
    eligible = raw[raw["label_source"] == "future_return_pending_threshold"].copy()
    eligible = temporal_resplit(eligible, train_ratio=train_ratio, val_ratio=val_ratio)
    labeled = assign_future_return_threshold(
        eligible,
        top_percentile=config.top_percentile,
        config=config,
    )
    return labeled[labeled["Trader_Success_Rate"] != -1].copy()


def run_label_sensitivity(
    observation_days: Sequence[int] = (1,),
    horizon_days: Sequence[int] = (1, 3, 7, 14),
    min_trades_values: Sequence[int] = (1, 3),
    top_percentiles: Sequence[float] = (0.1, 0.2),
    min_test_rows: int = 50,
    min_test_positive: int = 5,
    trades_df: pd.DataFrame | None = None,
    resolutions: Dict[str, str] | None = None,
) -> Dict:
    """Run Phase 8/10 label coverage sensitivity over future-return configs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if trades_df is None:
        trades_df = pd.read_csv(DATA_DIR / "processed" / "cleaned_transactions.csv")
    if resolutions is None:
        resolutions = load_resolutions()

    rows: List[Dict] = []
    for obs in observation_days:
        for horizon in horizon_days:
            for min_trades in min_trades_values:
                for top_pct in top_percentiles:
                    config = FutureReturnLabelConfig(
                        observation_days=int(obs),
                        horizon_days=int(horizon),
                        min_future_resolved_trades=int(min_trades),
                        top_percentile=float(top_pct),
                    )
                    labeled = build_future_label_frame(trades_df, resolutions, config)
                    summary = summarise_labeled_frame(labeled)
                    row = {
                        **asdict(config),
                        **summary,
                        "meets_minimum_test_gate": (
                            not summary["single_class_splits"]
                            and summary["test_rows"] >= min_test_rows
                            and summary["test_positive"] >= min_test_positive
                        ),
                    }
                    rows.append(row)

    result_df = pd.DataFrame(rows)
    result_path = RESULTS_DIR / "future_label_sensitivity.csv"
    result_df.to_csv(result_path, index=False)

    candidates = result_df[result_df["meets_minimum_test_gate"]].copy()
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["test_rows", "test_positive", "rows"],
            ascending=[False, False, False],
        )
        recommended = candidates.iloc[0].to_dict()
    elif not result_df.empty:
        fallback = result_df.sort_values(["test_rows", "test_positive", "rows"], ascending=False).iloc[0]
        recommended = fallback.to_dict()
    else:
        recommended = {}

    report = {
        "status": "ok" if rows else "no_configs",
        "config_count": len(rows),
        "minimum_test_gate": {
            "min_test_rows": min_test_rows,
            "min_test_positive": min_test_positive,
        },
        "recommended_config": recommended,
        "recommendation_passes_gate": bool(recommended.get("meets_minimum_test_gate", False)),
        "output_csv": str(result_path),
    }
    report_path = RESULTS_DIR / "future_label_sensitivity_summary.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info(f"Wrote future-label sensitivity → {result_path}")
    logger.info(f"Wrote future-label sensitivity summary → {report_path}")
    return report


def _positive_proba(model, x):
    proba = model.predict_proba(x)
    classes = list(getattr(model, "classes_", [0, 1]))
    return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(x))


def _precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    top_k = min(k, len(y_true))
    order = np.argsort(-y_prob)[:top_k]
    return float(np.mean(y_true[order] == 1))


def _threshold_policy_rows(
    fold: int,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    top_k_values: Sequence[int],
    top_percentiles: Sequence[float],
) -> List[Dict]:
    """Evaluate threshold/selection policies without assuming calibrated 0.5 probabilities."""
    rows: List[Dict] = []
    policies: List[tuple[str, np.ndarray]] = [
        ("fixed_0_5", y_prob >= 0.5),
    ]
    order = np.argsort(-y_prob)
    for k in top_k_values:
        mask = np.zeros(len(y_prob), dtype=bool)
        mask[order[: min(int(k), len(order))]] = True
        policies.append((f"top_k_{int(k)}", mask))
    for pct in top_percentiles:
        n = max(1, int(np.ceil(len(y_prob) * float(pct)))) if len(y_prob) else 0
        mask = np.zeros(len(y_prob), dtype=bool)
        mask[order[:n]] = True
        policies.append((f"top_pct_{float(pct):.2f}", mask))

    positives = int((y_true == 1).sum())
    for policy, mask in policies:
        selected = int(mask.sum())
        true_positive = int(((y_true == 1) & mask).sum())
        rows.append({
            "fold": fold,
            "policy": policy,
            "selected": selected,
            "support": int(len(y_true)),
            "positives": positives,
            "true_positive": true_positive,
            "precision": true_positive / selected if selected else 0.0,
            "recall": true_positive / positives if positives else 0.0,
        })
    return rows


def make_rolling_origin_folds(df: pd.DataFrame, n_folds: int = 5, min_train_fraction: float = 0.4) -> List[Dict]:
    """Create cumulative train / next-block test folds from first_trade_date order."""
    ordered = df.copy()
    ordered["first_trade_date"] = _normalise_time(ordered["first_trade_date"])
    ordered = ordered.dropna(subset=["first_trade_date"]).sort_values(
        ["first_trade_date", "address"], kind="mergesort"
    )
    if ordered.empty:
        return []

    min_train = max(1, int(len(ordered) * min_train_fraction))
    remaining = ordered.iloc[min_train:].copy()
    if remaining.empty:
        return []

    chunks = np.array_split(remaining.index.to_numpy(), n_folds)
    folds = []
    for fold_idx, test_idx in enumerate(chunks, start=1):
        if len(test_idx) == 0:
            continue
        test_start_pos = ordered.index.get_loc(test_idx[0])
        train = ordered.iloc[:test_start_pos].copy()
        test = ordered.loc[test_idx].copy()
        folds.append({"fold": fold_idx, "train": train, "test": test})
    return folds


def run_rolling_origin_evaluation(
    feature_path: Path | None = None,
    n_folds: int = 5,
    min_train_fraction: float = 0.4,
    feature_set: str = "live",
    top_k: int = 20,
    policy_top_k_values: Sequence[int] = (10, 20, 50),
    policy_top_percentiles: Sequence[float] = (0.05, 0.10, 0.20),
) -> Dict:
    """Run Phase 9 rolling-origin evaluation with a fast calibrated-free LR model."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = feature_path or (DATA_DIR / "features" / "model_input.csv")
    df = pd.read_csv(feature_path)
    feature_cols = get_feature_columns(feature_set)
    folds = make_rolling_origin_folds(df, n_folds=n_folds, min_train_fraction=min_train_fraction)

    rows = []
    prediction_rows = []
    policy_rows = []
    for fold in folds:
        train = fold["train"]
        test = fold["test"]
        y_train = train["Trader_Success_Rate"].astype(int).to_numpy()
        y_test = test["Trader_Success_Rate"].astype(int).to_numpy()
        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)
        row = {
            "fold": fold["fold"],
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_positive": int((y_train == 1).sum()),
            "test_positive": int((y_test == 1).sum()),
            "train_start": str(train["first_trade_date"].min()),
            "train_end": str(train["first_trade_date"].max()),
            "test_start": str(test["first_trade_date"].min()),
            "test_end": str(test["first_trade_date"].max()),
            "status": "ok",
        }
        if len(train_classes) < 2 or len(test_classes) < 2:
            row.update({
                "status": "single_class_fold",
                "accuracy": np.nan,
                "auc_roc": np.nan,
                "avg_precision": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "precision_at_k": np.nan,
            })
            rows.append(row)
            continue

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ])
        model.fit(train[feature_cols].to_numpy(), y_train)
        y_prob = _positive_proba(model, test[feature_cols].to_numpy())
        y_pred = (y_prob >= 0.5).astype(int)
        ranked = test[["address", "first_trade_date", "Trader_Success_Rate"]].copy()
        ranked["fold"] = fold["fold"]
        ranked["predicted_probability"] = y_prob
        ranked["predicted_label_0_5"] = y_pred
        ranked["rank_in_fold"] = ranked["predicted_probability"].rank(
            method="first", ascending=False
        ).astype(int)
        ranked["test_start"] = row["test_start"]
        ranked["test_end"] = row["test_end"]
        prediction_rows.extend(ranked.to_dict("records"))
        policy_rows.extend(
            _threshold_policy_rows(
                fold=fold["fold"],
                y_true=y_test,
                y_prob=y_prob,
                top_k_values=policy_top_k_values,
                top_percentiles=policy_top_percentiles,
            )
        )
        row.update({
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "auc_roc": float(roc_auc_score(y_test, y_prob)),
            "avg_precision": float(average_precision_score(y_test, y_prob)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "precision_at_k": _precision_at_k(y_test, y_prob, top_k),
        })
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_path = RESULTS_DIR / "rolling_origin_results.csv"
    result_df.to_csv(result_path, index=False)
    predictions_df = pd.DataFrame(prediction_rows)
    predictions_path = RESULTS_DIR / "rolling_origin_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    policy_df = pd.DataFrame(policy_rows)
    policy_path = RESULTS_DIR / "threshold_policy_results.csv"
    policy_df.to_csv(policy_path, index=False)
    ok = result_df[result_df["status"] == "ok"] if not result_df.empty else pd.DataFrame()
    policy_summary = {}
    if not policy_df.empty:
        policy_summary = {
            policy: {
                "mean_precision": float(group["precision"].mean()),
                "mean_recall": float(group["recall"].mean()),
                "mean_selected": float(group["selected"].mean()),
                "total_true_positive": int(group["true_positive"].sum()),
            }
            for policy, group in policy_df.groupby("policy")
        }
    summary = {
        "status": "ok" if not ok.empty else "no_valid_folds",
        "fold_count": int(len(result_df)),
        "valid_fold_count": int(len(ok)),
        "feature_set": feature_set,
        "top_k": top_k,
        "aggregate": {
            "mean_auc_roc": float(ok["auc_roc"].mean()) if not ok.empty else None,
            "mean_avg_precision": float(ok["avg_precision"].mean()) if not ok.empty else None,
            "mean_precision_at_k": float(ok["precision_at_k"].mean()) if not ok.empty else None,
            "total_test_rows": int(ok["test_rows"].sum()) if not ok.empty else 0,
            "total_test_positive": int(ok["test_positive"].sum()) if not ok.empty else 0,
        },
        "threshold_policies": policy_summary,
        "output_csv": str(result_path),
        "predictions_csv": str(predictions_path),
        "threshold_policy_csv": str(policy_path),
    }
    summary_path = RESULTS_DIR / "rolling_origin_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Wrote rolling-origin results → {result_path}")
    logger.info(f"Wrote rolling-origin predictions → {predictions_path}")
    logger.info(f"Wrote threshold policy results → {policy_path}")
    logger.info(f"Wrote rolling-origin summary → {summary_path}")
    return summary


def _select_top_volume_targets(tx_df: pd.DataFrame, fold_start: pd.Timestamp, max_targets: int) -> List[str]:
    history = tx_df[_normalise_time(tx_df["timestamp"]) < fold_start].copy()
    if history.empty:
        return []
    history["volume"] = history.get("volume", history["amount"] * history["price"])
    ranked = history.groupby("address")["volume"].sum().sort_values(ascending=False)
    return ranked.head(max_targets).index.tolist()


def _select_random_targets(
    tx_df: pd.DataFrame,
    fold_start: pd.Timestamp,
    max_targets: int,
    seed: int,
) -> List[str]:
    history = tx_df[_normalise_time(tx_df["timestamp"]) < fold_start]
    addresses = sorted(history["address"].dropna().unique().tolist())
    if not addresses:
        return []
    rng = np.random.default_rng(seed)
    return rng.choice(addresses, size=min(max_targets, len(addresses)), replace=False).tolist()


def run_model_informed_walk_forward(
    predictions_path: Path | None = None,
    top_k: int = 20,
    top_percentile: float = 0.10,
    latency_minutes: int = 5,
    trade_size: float = 100.0,
    fees_pct: float = 0.001,
    random_seed: int = 42,
) -> Dict:
    """Backtest model-selected wallets from rolling-origin out-of-sample predictions."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_path or (RESULTS_DIR / "rolling_origin_predictions.csv")
    if not predictions_path.exists():
        return {"status": "missing_predictions", "predictions_csv": str(predictions_path)}

    pred_df = pd.read_csv(predictions_path)
    if pred_df.empty:
        return {"status": "empty_predictions", "predictions_csv": str(predictions_path)}

    backtester = StrategyBacktester(
        latency_minutes=latency_minutes,
        trade_size=trade_size,
        fees_pct=fees_pct,
    )
    backtester.load_data()

    rows = []
    fold_details = []
    for fold, group in pred_df.groupby("fold"):
        group = group.sort_values("predicted_probability", ascending=False)
        start = pd.Timestamp(group["test_start"].iloc[0])
        end = pd.Timestamp(group["test_end"].iloc[0])
        top_pct_n = max(1, int(np.ceil(len(group) * top_percentile)))
        strategies = {
            f"model_top_k_{top_k}": group.head(top_k)["address"].tolist(),
            f"model_top_pct_{top_percentile:.2f}": group.head(top_pct_n)["address"].tolist(),
            "baseline_top_volume": _select_top_volume_targets(backtester.tx_df, start, top_k),
            "baseline_random_wallets": _select_random_targets(
                backtester.tx_df, start, top_k, random_seed + int(fold)
            ),
        }
        fold_details.append({
            "fold": int(fold),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "target_counts": {name: len(targets) for name, targets in strategies.items()},
        })
        for strategy_name, targets in strategies.items():
            metrics = backtester.simulate_for_addresses(
                target_addresses=targets,
                strategy_name=strategy_name,
                window_start=start,
                window_end=end,
                output_prefix=None,
            )
            rows.append({
                "fold": int(fold),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "strategy": strategy_name,
                "target_count": len(targets),
                "status": metrics.get("status"),
                "total_trades": metrics.get("Total_Trades", 0),
                "requested_trades": metrics.get("Requested_Trades", 0),
                "total_requested_notional": metrics.get("Total_Requested_Notional", 0.0),
                "total_filled_notional": metrics.get("Total_Filled_Notional", 0.0),
                "fill_rate": metrics.get("Fill_Rate", 0.0),
                "skipped_liquidity": metrics.get("Skipped_Liquidity", 0),
                "skipped_capital": metrics.get("Skipped_Capital", 0),
                "capital_capped_trades": metrics.get("Capital_Capped_Trades", 0),
                "liquidity_capped_trades": metrics.get("Liquidity_Capped_Trades", 0),
                "market_exposure_capped_trades": metrics.get("Market_Exposure_Capped_Trades", 0),
                "win_rate": metrics.get("Win_Rate", 0.0),
                "total_net_profit": metrics.get("Total_Net_Profit", 0.0),
                "roi_pct": metrics.get("ROI_Pct", 0.0),
                "max_drawdown": metrics.get("Max_Drawdown", 0.0),
                "sharpe": metrics.get("Sharpe_Ratio (Annualized)", 0.0),
                "unique_markets": metrics.get("Unique_Markets", 0),
                "top_market_share": metrics.get("Top_Market_Filled_Notional_Share", 0.0),
                "market_hhi": metrics.get("Market_HHI", 0.0),
                "effective_markets": metrics.get("Effective_Markets", 0.0),
                "reason": metrics.get("reason", ""),
            })

    result_df = add_equal_capital_metrics(pd.DataFrame(rows))
    result_path = RESULTS_DIR / "model_informed_walk_forward_results.csv"
    result_df.to_csv(result_path, index=False)
    summary = {
        "status": "ok" if rows else "no_folds",
        "predictions_csv": str(predictions_path),
        "top_k": top_k,
        "top_percentile": top_percentile,
        "folds": fold_details,
        "equal_capital_method": EQUAL_CAPITAL_METHOD,
        "aggregate": {},
        "output_csv": str(result_path),
    }
    if not result_df.empty:
        summary["aggregate"] = {
            strategy: {
                "folds": int(len(group)),
                "total_trades": int(group["total_trades"].sum()),
                "requested_trades": int(group["requested_trades"].sum()),
                "total_filled_notional": float(group["total_filled_notional"].sum()),
                "mean_fill_rate": float(group["fill_rate"].mean()),
                "total_net_profit": float(group["total_net_profit"].sum()),
                "mean_roi_pct": float(group["roi_pct"].mean()),
                "total_equal_capital_budget": float(group["equal_capital_budget"].sum()),
                "total_equal_capital_net_profit": float(group["equal_capital_net_profit"].sum()),
                "mean_equal_capital_roi": float(group["equal_capital_roi"].mean()),
                "equal_capital_supported_folds": int(group["equal_capital_supported"].sum()),
                "mean_sharpe": float(group["sharpe"].mean()),
                "mean_unique_markets": float(group["unique_markets"].mean()),
                "mean_top_market_share": float(group["top_market_share"].mean()),
                "mean_market_hhi": float(group["market_hhi"].mean()),
                "mean_effective_markets": float(group["effective_markets"].mean()),
            }
            for strategy, group in result_df.groupby("strategy")
        }
    summary_path = RESULTS_DIR / "model_informed_walk_forward_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Wrote model-informed walk-forward results → {result_path}")
    logger.info(f"Wrote model-informed walk-forward summary → {summary_path}")
    return summary


def _numeric_distribution(series: pd.Series) -> Dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "p05": float(values.quantile(0.05)),
        "p50": float(values.quantile(0.50)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def _random_baseline_comparison(random_values: pd.Series, observed_value: float | None) -> Dict:
    values = pd.to_numeric(random_values, errors="coerce").dropna()
    if observed_value is None or values.empty:
        return {"status": "insufficient_data"}
    observed = float(observed_value)
    percentile = float((values <= observed).mean())
    right_tail_p = float(((values >= observed).sum() + 1) / (len(values) + 1))
    return {
        "status": "ok",
        "observed": observed,
        "random_percentile": percentile,
        "right_tail_p_value": right_tail_p,
        "random_seed_count": int(len(values)),
    }


def _confidence_interval(values: pd.Series, lower: float = 0.025, upper: float = 0.975) -> Dict:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"status": "insufficient_data", "count": 0}
    return {
        "status": "ok",
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "p025": float(numeric.quantile(lower)),
        "p50": float(numeric.quantile(0.50)),
        "p975": float(numeric.quantile(upper)),
    }


def _distribution_position(random_values: pd.Series, observed_value: float | None) -> Dict:
    values = pd.to_numeric(random_values, errors="coerce").dropna()
    if observed_value is None or values.empty:
        return {"status": "insufficient_data"}
    observed = float(observed_value)
    return {
        "status": "ok",
        "observed": observed,
        "random_percentile": float((values <= observed).mean()),
        "left_tail_p_value": float(((values <= observed).sum() + 1) / (len(values) + 1)),
        "right_tail_p_value": float(((values >= observed).sum() + 1) / (len(values) + 1)),
        "random_seed_count": int(len(values)),
    }


def run_random_baseline_stability(
    predictions_path: Path | None = None,
    model_summary_path: Path | None = None,
    seed_count: int = 10,
    top_k: int = 20,
    top_percentile: float = 0.10,
    latency_minutes: int = 5,
    trade_size: float = 100.0,
    fees_pct: float = 0.001,
    random_seed: int = 10_000,
) -> Dict:
    """Run multi-seed random-wallet walk-forward baselines for stability checks."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_path or (RESULTS_DIR / "rolling_origin_predictions.csv")
    model_summary_path = model_summary_path or (RESULTS_DIR / "model_informed_walk_forward_summary.json")
    if not predictions_path.exists():
        return {"status": "missing_predictions", "predictions_csv": str(predictions_path)}

    pred_df = pd.read_csv(predictions_path)
    if pred_df.empty:
        return {"status": "empty_predictions", "predictions_csv": str(predictions_path)}

    backtester = StrategyBacktester(
        latency_minutes=latency_minutes,
        trade_size=trade_size,
        fees_pct=fees_pct,
    )
    backtester.load_data()

    rows = []
    for fold, group in pred_df.groupby("fold"):
        start = pd.Timestamp(group["test_start"].iloc[0])
        end = pd.Timestamp(group["test_end"].iloc[0])
        for seed_idx in range(seed_count):
            seed = random_seed + seed_idx + int(fold) * 10_000
            targets = _select_random_targets(backtester.tx_df, start, top_k, seed)
            metrics = backtester.simulate_for_addresses(
                target_addresses=targets,
                strategy_name="baseline_random_wallets",
                window_start=start,
                window_end=end,
                output_prefix=None,
            )
            rows.append({
                "fold": int(fold),
                "seed_index": int(seed_idx),
                "seed": int(seed),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "strategy": "baseline_random_wallets",
                "target_count": len(targets),
                "status": metrics.get("status"),
                "total_trades": metrics.get("Total_Trades", 0),
                "requested_trades": metrics.get("Requested_Trades", 0),
                "total_requested_notional": metrics.get("Total_Requested_Notional", 0.0),
                "total_filled_notional": metrics.get("Total_Filled_Notional", 0.0),
                "fill_rate": metrics.get("Fill_Rate", 0.0),
                "skipped_liquidity": metrics.get("Skipped_Liquidity", 0),
                "skipped_capital": metrics.get("Skipped_Capital", 0),
                "capital_capped_trades": metrics.get("Capital_Capped_Trades", 0),
                "liquidity_capped_trades": metrics.get("Liquidity_Capped_Trades", 0),
                "market_exposure_capped_trades": metrics.get("Market_Exposure_Capped_Trades", 0),
                "win_rate": metrics.get("Win_Rate", 0.0),
                "total_net_profit": metrics.get("Total_Net_Profit", 0.0),
                "roi_pct": metrics.get("ROI_Pct", 0.0),
                "max_drawdown": metrics.get("Max_Drawdown", 0.0),
                "sharpe": metrics.get("Sharpe_Ratio (Annualized)", 0.0),
                "unique_markets": metrics.get("Unique_Markets", 0),
                "top_market_share": metrics.get("Top_Market_Filled_Notional_Share", 0.0),
                "market_hhi": metrics.get("Market_HHI", 0.0),
                "effective_markets": metrics.get("Effective_Markets", 0.0),
                "reason": metrics.get("reason", ""),
            })

    result_df = add_equal_capital_metrics(pd.DataFrame(rows))
    result_path = RESULTS_DIR / "random_baseline_stability.csv"
    result_df.to_csv(result_path, index=False)

    seed_summary_df = pd.DataFrame()
    if not result_df.empty:
        seed_summary_df = pd.DataFrame([
            {
                "seed_index": int(seed_idx),
                "folds": int(len(group)),
                "supported_folds": int(group["equal_capital_supported"].sum()),
                "total_trades": int(group["total_trades"].sum()),
                "requested_trades": int(group["requested_trades"].sum()),
                "total_filled_notional": float(group["total_filled_notional"].sum()),
                "mean_fill_rate": float(group["fill_rate"].mean()),
                "total_net_profit": float(group["total_net_profit"].sum()),
                "mean_roi_pct": float(group["roi_pct"].mean()),
                "total_equal_capital_budget": float(group["equal_capital_budget"].sum()),
                "total_equal_capital_net_profit": float(group["equal_capital_net_profit"].sum()),
                "mean_equal_capital_roi": float(group["equal_capital_roi"].mean()),
                "mean_sharpe": float(group["sharpe"].mean()),
                "mean_unique_markets": float(group["unique_markets"].mean()),
                "mean_top_market_share": float(group["top_market_share"].mean()),
                "mean_market_hhi": float(group["market_hhi"].mean()),
                "mean_effective_markets": float(group["effective_markets"].mean()),
            }
            for seed_idx, group in result_df.groupby("seed_index")
        ])
    seed_summary_path = RESULTS_DIR / "random_baseline_seed_summary.csv"
    seed_summary_df.to_csv(seed_summary_path, index=False)

    model_summary = _read_json(model_summary_path) if model_summary_path.exists() else {}
    model_strategy = f"model_top_pct_{top_percentile:.2f}"
    model_metrics = model_summary.get("aggregate", {}).get(model_strategy, {})
    comparisons = {
        "model_strategy": model_strategy,
        "total_net_profit": _random_baseline_comparison(
            seed_summary_df.get("total_net_profit", pd.Series(dtype=float)),
            model_metrics.get("total_net_profit"),
        ),
        "mean_equal_capital_roi": _random_baseline_comparison(
            seed_summary_df.get("mean_equal_capital_roi", pd.Series(dtype=float)),
            model_metrics.get("mean_equal_capital_roi"),
        ),
        "total_equal_capital_net_profit": _random_baseline_comparison(
            seed_summary_df.get("total_equal_capital_net_profit", pd.Series(dtype=float)),
            model_metrics.get("total_equal_capital_net_profit"),
        ),
    }
    summary = {
        "status": "ok" if rows else "no_folds",
        "predictions_csv": str(predictions_path),
        "seed_count": int(seed_count),
        "top_k": int(top_k),
        "top_percentile": float(top_percentile),
        "random_seed": int(random_seed),
        "distributions": {
            "total_net_profit": _numeric_distribution(seed_summary_df.get("total_net_profit", pd.Series(dtype=float))),
            "mean_equal_capital_roi": _numeric_distribution(seed_summary_df.get("mean_equal_capital_roi", pd.Series(dtype=float))),
            "total_equal_capital_net_profit": _numeric_distribution(
                seed_summary_df.get("total_equal_capital_net_profit", pd.Series(dtype=float))
            ),
            "mean_fill_rate": _numeric_distribution(seed_summary_df.get("mean_fill_rate", pd.Series(dtype=float))),
            "mean_market_hhi": _numeric_distribution(seed_summary_df.get("mean_market_hhi", pd.Series(dtype=float))),
            "mean_top_market_share": _numeric_distribution(
                seed_summary_df.get("mean_top_market_share", pd.Series(dtype=float))
            ),
            "mean_effective_markets": _numeric_distribution(
                seed_summary_df.get("mean_effective_markets", pd.Series(dtype=float))
            ),
        },
        "model_vs_random": comparisons,
        "output_csv": str(result_path),
        "seed_summary_csv": str(seed_summary_path),
    }
    summary_path = RESULTS_DIR / "random_baseline_stability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Wrote random baseline stability rows → {result_path}")
    logger.info(f"Wrote random baseline seed summary → {seed_summary_path}")
    logger.info(f"Wrote random baseline stability summary → {summary_path}")
    return summary


def run_exposure_concentration_analysis(
    model_summary_path: Path | None = None,
    random_seed_summary_path: Path | None = None,
    top_percentile: float = 0.10,
) -> Dict:
    """Compare model strategy market concentration against random baseline seeds."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_summary_path = model_summary_path or (RESULTS_DIR / "model_informed_walk_forward_summary.json")
    random_seed_summary_path = random_seed_summary_path or (RESULTS_DIR / "random_baseline_seed_summary.csv")
    model_summary = _read_json(model_summary_path)
    if not model_summary:
        return {"status": "missing_model_summary", "model_summary_json": str(model_summary_path)}
    if not random_seed_summary_path.exists():
        return {"status": "missing_random_seed_summary", "random_seed_summary_csv": str(random_seed_summary_path)}

    seed_summary_df = pd.read_csv(random_seed_summary_path)
    model_strategy = f"model_top_pct_{top_percentile:.2f}"
    model_metrics = model_summary.get("aggregate", {}).get(model_strategy, {})
    concentration_metrics = [
        "mean_unique_markets",
        "mean_top_market_share",
        "mean_market_hhi",
        "mean_effective_markets",
    ]
    summary = {
        "status": "ok",
        "model_strategy": model_strategy,
        "model_summary_json": str(model_summary_path),
        "random_seed_summary_csv": str(random_seed_summary_path),
        "interpretation": {
            "mean_market_hhi": "lower means less market concentration",
            "mean_top_market_share": "lower means less dependence on one market",
            "mean_effective_markets": "higher means broader market exposure",
        },
        "random_distributions": {
            metric: _numeric_distribution(seed_summary_df.get(metric, pd.Series(dtype=float)))
            for metric in concentration_metrics
        },
        "model_vs_random": {
            metric: _distribution_position(
                seed_summary_df.get(metric, pd.Series(dtype=float)),
                model_metrics.get(metric),
            )
            for metric in concentration_metrics
        },
    }
    summary_path = RESULTS_DIR / "exposure_concentration_summary.json"
    summary["output_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Wrote exposure concentration summary → {summary_path}")
    return summary


def _standardized_distance(row: pd.Series, observed: Dict, metrics: Sequence[str], scales: Dict[str, float]) -> float:
    squared = []
    for metric in metrics:
        value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
        target = observed.get(metric)
        if pd.isna(value) or target is None:
            continue
        scale = scales.get(metric, 1.0) or 1.0
        squared.append(((float(value) - float(target)) / scale) ** 2)
    if not squared:
        return float("inf")
    return float(np.sqrt(np.mean(squared)))


def run_stratified_random_baseline_analysis(
    model_summary_path: Path | None = None,
    random_seed_summary_path: Path | None = None,
    top_percentile: float = 0.10,
    match_count: int | None = None,
) -> Dict:
    """Compare model performance against random seeds matched on execution profile."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_summary_path = model_summary_path or (RESULTS_DIR / "model_informed_walk_forward_summary.json")
    random_seed_summary_path = random_seed_summary_path or (RESULTS_DIR / "random_baseline_seed_summary.csv")
    model_summary = _read_json(model_summary_path)
    if not model_summary:
        return {"status": "missing_model_summary", "model_summary_json": str(model_summary_path)}
    if not random_seed_summary_path.exists():
        return {"status": "missing_random_seed_summary", "random_seed_summary_csv": str(random_seed_summary_path)}

    seed_summary_df = pd.read_csv(random_seed_summary_path)
    if seed_summary_df.empty:
        return {"status": "empty_random_seed_summary", "random_seed_summary_csv": str(random_seed_summary_path)}

    model_strategy = f"model_top_pct_{top_percentile:.2f}"
    model_metrics = model_summary.get("aggregate", {}).get(model_strategy, {})
    control_metrics = [
        "total_trades",
        "mean_fill_rate",
        "mean_market_hhi",
        "mean_top_market_share",
        "mean_effective_markets",
    ]
    missing_controls = [
        metric
        for metric in control_metrics
        if metric not in seed_summary_df.columns or model_metrics.get(metric) is None
    ]
    if missing_controls:
        return {
            "status": "missing_control_metrics",
            "missing_controls": missing_controls,
            "model_strategy": model_strategy,
        }

    scales = {}
    for metric in control_metrics:
        values = pd.to_numeric(seed_summary_df[metric], errors="coerce").dropna()
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        scales[metric] = std if std > 0 else 1.0

    scored = seed_summary_df.copy()
    scored["match_distance"] = scored.apply(
        lambda row: _standardized_distance(row, model_metrics, control_metrics, scales),
        axis=1,
    )
    scored = scored.sort_values(["match_distance", "seed_index"], kind="mergesort")
    default_match_count = max(3, int(np.ceil(len(scored) * 0.5)))
    selected_count = min(len(scored), int(match_count or default_match_count))
    matched = scored.head(selected_count).copy()

    performance_metrics = [
        "total_net_profit",
        "mean_equal_capital_roi",
        "total_equal_capital_net_profit",
    ]
    comparisons = {
        "model_strategy": model_strategy,
        **{
            metric: _random_baseline_comparison(
                matched.get(metric, pd.Series(dtype=float)),
                model_metrics.get(metric),
            )
            for metric in performance_metrics
        },
    }
    control_balance = {
        metric: {
            "model": float(model_metrics.get(metric)),
            "matched_random_mean": float(pd.to_numeric(matched[metric], errors="coerce").mean()),
            "all_random_mean": float(pd.to_numeric(scored[metric], errors="coerce").mean()),
            "scale": float(scales[metric]),
        }
        for metric in control_metrics
    }

    matches_path = RESULTS_DIR / "stratified_random_baseline_matches.csv"
    matched.to_csv(matches_path, index=False)
    summary = {
        "status": "ok",
        "model_strategy": model_strategy,
        "model_summary_json": str(model_summary_path),
        "random_seed_summary_csv": str(random_seed_summary_path),
        "match_method": "nearest_random_seeds_by_standardized_execution_profile_distance",
        "control_metrics": control_metrics,
        "available_seed_count": int(len(scored)),
        "matched_seed_count": int(len(matched)),
        "matched_seed_indices": [int(seed) for seed in matched["seed_index"].tolist()],
        "mean_match_distance": float(matched["match_distance"].mean()),
        "control_balance": control_balance,
        "matched_distributions": {
            metric: _numeric_distribution(matched.get(metric, pd.Series(dtype=float)))
            for metric in performance_metrics
        },
        "model_vs_matched_random": comparisons,
        "matches_csv": str(matches_path),
    }
    summary_path = RESULTS_DIR / "stratified_random_baseline_summary.json"
    summary["output_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Wrote stratified random baseline matches → {matches_path}")
    logger.info(f"Wrote stratified random baseline summary → {summary_path}")
    return summary


def _aggregate_fold_rows(rows: pd.DataFrame, control_metrics: Sequence[str]) -> Dict:
    if rows.empty:
        return {}
    aggregate = {
        "folds": int(rows["fold"].nunique()) if "fold" in rows.columns else int(len(rows)),
        "total_trades": int(pd.to_numeric(rows.get("total_trades", pd.Series(dtype=float)), errors="coerce").sum()),
        "requested_trades": int(pd.to_numeric(rows.get("requested_trades", pd.Series(dtype=float)), errors="coerce").sum()),
        "total_filled_notional": float(pd.to_numeric(rows.get("total_filled_notional", pd.Series(dtype=float)), errors="coerce").sum()),
        "mean_fill_rate": float(pd.to_numeric(rows.get("fill_rate", pd.Series(dtype=float)), errors="coerce").mean()),
        "total_net_profit": float(pd.to_numeric(rows.get("total_net_profit", pd.Series(dtype=float)), errors="coerce").sum()),
        "mean_roi_pct": float(pd.to_numeric(rows.get("roi_pct", pd.Series(dtype=float)), errors="coerce").mean()),
        "total_equal_capital_net_profit": float(
            pd.to_numeric(rows.get("equal_capital_net_profit", pd.Series(dtype=float)), errors="coerce").sum()
        ),
        "mean_equal_capital_roi": float(
            pd.to_numeric(rows.get("equal_capital_roi", pd.Series(dtype=float)), errors="coerce").mean()
        ),
        "mean_market_hhi": float(pd.to_numeric(rows.get("market_hhi", pd.Series(dtype=float)), errors="coerce").mean()),
        "mean_top_market_share": float(
            pd.to_numeric(rows.get("top_market_share", pd.Series(dtype=float)), errors="coerce").mean()
        ),
        "mean_effective_markets": float(
            pd.to_numeric(rows.get("effective_markets", pd.Series(dtype=float)), errors="coerce").mean()
        ),
    }
    for metric in control_metrics:
        if metric not in aggregate and metric in rows.columns:
            aggregate[metric] = float(pd.to_numeric(rows[metric], errors="coerce").mean())
    return aggregate


def _bootstrap_fold_matched_differences(
    model_rows: pd.DataFrame,
    matches_df: pd.DataFrame,
    performance_metrics: Sequence[str],
    control_metrics: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict:
    """Resample folds and matched random candidates to estimate model-minus-random intervals."""
    fold_ids = sorted(set(model_rows["fold"].astype(int)).intersection(matches_df["fold"].astype(int)))
    if not fold_ids or bootstrap_samples <= 0:
        return {"status": "insufficient_data", "fold_count": int(len(fold_ids))}

    model_by_fold = {
        int(fold): group.copy()
        for fold, group in model_rows.groupby(model_rows["fold"].astype(int))
    }
    random_by_fold = {
        int(fold): group.copy()
        for fold, group in matches_df.groupby(matches_df["fold"].astype(int))
    }
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for sample_index in range(int(bootstrap_samples)):
        sampled_folds = rng.choice(fold_ids, size=len(fold_ids), replace=True)
        sampled_model_rows = []
        sampled_random_rows = []
        for fold in sampled_folds:
            fold = int(fold)
            sampled_model_rows.append(model_by_fold[fold])
            candidates = random_by_fold[fold]
            candidate = candidates.iloc[[int(rng.integers(0, len(candidates)))]]
            sampled_random_rows.append(candidate)
        model_aggregate = _aggregate_fold_rows(pd.concat(sampled_model_rows), control_metrics)
        random_aggregate = _aggregate_fold_rows(pd.concat(sampled_random_rows), control_metrics)
        row = {"bootstrap_sample": int(sample_index)}
        for metric in performance_metrics:
            model_value = model_aggregate.get(metric)
            random_value = random_aggregate.get(metric)
            if model_value is None or random_value is None:
                row[f"{metric}_model_minus_random"] = np.nan
            else:
                row[f"{metric}_model_minus_random"] = float(model_value) - float(random_value)
        rows.append(row)

    bootstrap_df = pd.DataFrame(rows)
    bootstrap_path = RESULTS_DIR / "fold_matched_random_bootstrap.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)
    intervals = {
        metric: _confidence_interval(bootstrap_df.get(f"{metric}_model_minus_random", pd.Series(dtype=float)))
        for metric in performance_metrics
    }
    for metric in performance_metrics:
        column = f"{metric}_model_minus_random"
        values = pd.to_numeric(bootstrap_df.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
        if not values.empty:
            intervals[metric]["probability_model_beats_matched_random"] = float((values > 0).mean())
    return {
        "status": "ok",
        "fold_count": int(len(fold_ids)),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "difference_metric": "model_minus_matched_random",
        "intervals": intervals,
        "output_csv": str(bootstrap_path),
    }


def run_fold_matched_random_baseline_analysis(
    model_results_path: Path | None = None,
    random_results_path: Path | None = None,
    top_percentile: float = 0.10,
    match_count_per_fold: int = 3,
    max_match_distance: float | None = None,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict:
    """Nearest-neighbor random baseline matching inside each out-of-sample fold."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_results_path = model_results_path or (RESULTS_DIR / "model_informed_walk_forward_results.csv")
    random_results_path = random_results_path or (RESULTS_DIR / "random_baseline_stability.csv")
    if not model_results_path.exists():
        return {"status": "missing_model_results", "model_results_csv": str(model_results_path)}
    if not random_results_path.exists():
        return {"status": "missing_random_results", "random_results_csv": str(random_results_path)}

    model_df = pd.read_csv(model_results_path)
    random_df = pd.read_csv(random_results_path)
    if model_df.empty or random_df.empty:
        return {
            "status": "empty_inputs",
            "model_results_csv": str(model_results_path),
            "random_results_csv": str(random_results_path),
        }

    model_strategy = f"model_top_pct_{top_percentile:.2f}"
    model_rows = model_df[model_df["strategy"] == model_strategy].copy()
    random_rows = random_df[random_df["strategy"] == "baseline_random_wallets"].copy()
    if model_rows.empty or random_rows.empty:
        return {
            "status": "missing_strategy_rows",
            "model_strategy": model_strategy,
            "model_rows": int(len(model_rows)),
            "random_rows": int(len(random_rows)),
        }

    control_metrics = [
        "total_trades",
        "fill_rate",
        "market_hhi",
        "top_market_share",
        "effective_markets",
    ]
    missing_controls = [
        metric
        for metric in control_metrics
        if metric not in model_rows.columns or metric not in random_rows.columns
    ]
    if missing_controls:
        return {"status": "missing_control_metrics", "missing_controls": missing_controls}

    matched_rows = []
    fold_details = []
    for _, model_row in model_rows.sort_values("fold").iterrows():
        fold = int(model_row["fold"])
        candidates = random_rows[random_rows["fold"] == fold].copy()
        if candidates.empty:
            fold_details.append({"fold": fold, "status": "no_random_candidates"})
            continue

        scales = {}
        for metric in control_metrics:
            values = pd.to_numeric(candidates[metric], errors="coerce").dropna()
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            scales[metric] = std if std > 0 else 1.0
        observed = model_row.to_dict()
        candidates["match_distance"] = candidates.apply(
            lambda row: _standardized_distance(row, observed, control_metrics, scales),
            axis=1,
        )
        sort_cols = ["match_distance"]
        if "seed_index" in candidates.columns:
            sort_cols.append("seed_index")
        candidates = candidates.sort_values(sort_cols, kind="mergesort")
        selected = candidates.head(max(1, min(int(match_count_per_fold), len(candidates)))).copy()
        selected["passes_caliper"] = True
        if max_match_distance is not None:
            selected["passes_caliper"] = selected["match_distance"] <= float(max_match_distance)
            caliper_selected = selected[selected["passes_caliper"]].copy()
            if not caliper_selected.empty:
                selected = caliper_selected
        selected["match_rank"] = range(1, len(selected) + 1)
        selected["model_strategy"] = model_strategy
        for metric in control_metrics:
            selected[f"model_{metric}"] = model_row.get(metric)
        selected["model_total_net_profit"] = model_row.get("total_net_profit")
        selected["model_equal_capital_roi"] = model_row.get("equal_capital_roi")
        matched_rows.append(selected)
        fold_details.append({
            "fold": fold,
            "status": "ok",
            "candidate_count": int(len(candidates)),
            "matched_count": int(len(selected)),
            "best_distance": float(selected["match_distance"].min()),
            "max_selected_distance": float(selected["match_distance"].max()),
            "passes_caliper": bool(selected["passes_caliper"].all()),
        })

    if not matched_rows:
        return {
            "status": "no_matches",
            "model_strategy": model_strategy,
            "folds": fold_details,
        }

    matches_df = pd.concat(matched_rows, ignore_index=True)
    model_aggregate = _aggregate_fold_rows(model_rows, control_metrics)
    rank_summary_rows = []
    for rank, group in matches_df.groupby("match_rank"):
        aggregate = _aggregate_fold_rows(group, control_metrics)
        aggregate["match_rank"] = int(rank)
        aggregate["mean_match_distance"] = float(group["match_distance"].mean())
        rank_summary_rows.append(aggregate)
    rank_summary_df = pd.DataFrame(rank_summary_rows).sort_values("match_rank")

    performance_metrics = [
        "total_net_profit",
        "mean_equal_capital_roi",
        "total_equal_capital_net_profit",
    ]
    bootstrap = _bootstrap_fold_matched_differences(
        model_rows=model_rows,
        matches_df=matches_df,
        performance_metrics=performance_metrics,
        control_metrics=control_metrics,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    comparisons = {
        "model_strategy": model_strategy,
        **{
            metric: _random_baseline_comparison(
                rank_summary_df.get(metric, pd.Series(dtype=float)),
                model_aggregate.get(metric),
            )
            for metric in performance_metrics
        },
    }
    control_balance = {
        metric: {
            "model": float(model_aggregate.get(metric)),
            "matched_random_mean": float(pd.to_numeric(rank_summary_df[metric], errors="coerce").mean()),
        }
        for metric in [
            "total_trades",
            "mean_fill_rate",
            "mean_market_hhi",
            "mean_top_market_share",
            "mean_effective_markets",
        ]
        if metric in rank_summary_df.columns and model_aggregate.get(metric) is not None
    }

    matches_path = RESULTS_DIR / "fold_matched_random_baseline_matches.csv"
    rank_summary_path = RESULTS_DIR / "fold_matched_random_baseline_rank_summary.csv"
    matches_df.to_csv(matches_path, index=False)
    rank_summary_df.to_csv(rank_summary_path, index=False)
    summary = {
        "status": "ok",
        "model_strategy": model_strategy,
        "model_results_csv": str(model_results_path),
        "random_results_csv": str(random_results_path),
        "match_method": "within_fold_nearest_random_rows_by_standardized_execution_profile_distance",
        "control_metrics": control_metrics,
        "match_count_per_fold": int(match_count_per_fold),
        "max_match_distance": float(max_match_distance) if max_match_distance is not None else None,
        "folds": fold_details,
        "model_aggregate": model_aggregate,
        "matched_rank_count": int(len(rank_summary_df)),
        "matched_rows": int(len(matches_df)),
        "caliper_passed_folds": int(sum(1 for fold in fold_details if fold.get("passes_caliper"))),
        "bootstrap_model_minus_matched_random": bootstrap,
        "control_balance": control_balance,
        "matched_rank_distributions": {
            metric: _numeric_distribution(rank_summary_df.get(metric, pd.Series(dtype=float)))
            for metric in performance_metrics
        },
        "model_vs_fold_matched_random": comparisons,
        "matches_csv": str(matches_path),
        "rank_summary_csv": str(rank_summary_path),
    }
    summary_path = RESULTS_DIR / "fold_matched_random_baseline_summary.json"
    summary["output_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Wrote fold-matched random baseline matches → {matches_path}")
    logger.info(f"Wrote fold-matched random baseline rank summary → {rank_summary_path}")
    logger.info(f"Wrote fold-matched random baseline summary → {summary_path}")
    return summary


def run_phase8_10(
    observation_days: Sequence[int] = (1,),
    horizon_days: Sequence[int] = (1, 3, 7, 14),
    min_trades_values: Sequence[int] = (1, 3),
    top_percentiles: Sequence[float] = (0.1, 0.2),
    rolling_folds: int = 5,
    run_model_backtest: bool = True,
    run_baseline_stability: bool = True,
    baseline_seed_count: int = 20,
    run_exposure_concentration: bool = True,
    run_stratified_random: bool = True,
    run_fold_matched_random: bool = True,
    fold_match_count_per_fold: int = 3,
    fold_match_max_distance: float | None = None,
    fold_match_bootstrap_samples: int = 1000,
    fold_match_bootstrap_seed: int = 42,
) -> Dict:
    """Run Phase 8-12 future-label, rolling-origin, threshold, and backtest evidence."""
    sensitivity = run_label_sensitivity(
        observation_days=observation_days,
        horizon_days=horizon_days,
        min_trades_values=min_trades_values,
        top_percentiles=top_percentiles,
    )
    rolling = run_rolling_origin_evaluation(n_folds=rolling_folds)
    model_walk_forward = (
        run_model_informed_walk_forward()
        if run_model_backtest
        else {"status": "skipped_by_request"}
    )
    baseline_stability = (
        run_random_baseline_stability(seed_count=baseline_seed_count)
        if run_model_backtest and run_baseline_stability
        else {"status": "skipped_by_request"}
    )
    exposure_concentration = (
        run_exposure_concentration_analysis()
        if run_model_backtest
        and run_baseline_stability
        and run_exposure_concentration
        and model_walk_forward.get("status") == "ok"
        and baseline_stability.get("status") == "ok"
        else {"status": "skipped_by_request"}
    )
    stratified_random = (
        run_stratified_random_baseline_analysis()
        if run_model_backtest
        and run_baseline_stability
        and run_stratified_random
        and model_walk_forward.get("status") == "ok"
        and baseline_stability.get("status") == "ok"
        else {"status": "skipped_by_request"}
    )
    fold_matched_random = (
        run_fold_matched_random_baseline_analysis(
            match_count_per_fold=fold_match_count_per_fold,
            max_match_distance=fold_match_max_distance,
            bootstrap_samples=fold_match_bootstrap_samples,
            bootstrap_seed=fold_match_bootstrap_seed,
        )
        if run_model_backtest
        and run_baseline_stability
        and run_fold_matched_random
        and model_walk_forward.get("status") == "ok"
        and baseline_stability.get("status") == "ok"
        else {"status": "skipped_by_request"}
    )
    report = {
        "status": "ok",
        "phase8_sample_expansion": sensitivity,
        "phase9_rolling_origin": rolling,
        "phase10_sensitivity": sensitivity,
        "phase11_threshold_policies": rolling.get("threshold_policies", {}),
        "phase12_model_informed_walk_forward": model_walk_forward,
        "phase14_equal_capital": {
            strategy: {
                key: metrics.get(key)
                for key in [
                    "total_equal_capital_budget",
                    "total_equal_capital_net_profit",
                    "mean_equal_capital_roi",
                    "equal_capital_supported_folds",
                ]
            }
            for strategy, metrics in model_walk_forward.get("aggregate", {}).items()
        },
        "phase15_random_baseline_stability": baseline_stability,
        "phase16_exposure_concentration": exposure_concentration,
        "phase17_stratified_random_baseline": stratified_random,
        "phase18_fold_matched_random_baseline": fold_matched_random,
    }
    legacy_report_path = RESULTS_DIR / "phase8_10_summary.json"
    current_report_path = RESULTS_DIR / "phase8_12_summary.json"
    for report_path in [legacy_report_path, current_report_path]:
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info(f"Wrote Phase 8-12 summary → {current_report_path}")
    return report


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run Phase 8-12 future-label experiments")
    parser.add_argument("--observation-days", default="1")
    parser.add_argument("--horizon-days", default="1,3,7,14")
    parser.add_argument("--min-trades", default="1,3")
    parser.add_argument("--top-percentiles", default="0.1,0.2")
    parser.add_argument("--rolling-folds", type=int, default=5)
    parser.add_argument("--skip-model-walk-forward", action="store_true")
    parser.add_argument("--skip-baseline-stability", action="store_true")
    parser.add_argument("--skip-exposure-concentration", action="store_true")
    parser.add_argument("--skip-stratified-random", action="store_true")
    parser.add_argument("--skip-fold-matched-random", action="store_true")
    parser.add_argument("--baseline-seed-count", type=int, default=20)
    parser.add_argument("--fold-match-count-per-fold", type=int, default=3)
    parser.add_argument("--fold-match-max-distance", type=float, default=None)
    parser.add_argument("--fold-match-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--fold-match-bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    run_phase8_10(
        observation_days=_parse_ints(args.observation_days),
        horizon_days=_parse_ints(args.horizon_days),
        min_trades_values=_parse_ints(args.min_trades),
        top_percentiles=_parse_floats(args.top_percentiles),
        rolling_folds=args.rolling_folds,
        run_model_backtest=not args.skip_model_walk_forward,
        run_baseline_stability=not args.skip_baseline_stability,
        baseline_seed_count=args.baseline_seed_count,
        run_exposure_concentration=not args.skip_exposure_concentration,
        run_stratified_random=not args.skip_stratified_random,
        run_fold_matched_random=not args.skip_fold_matched_random,
        fold_match_count_per_fold=args.fold_match_count_per_fold,
        fold_match_max_distance=args.fold_match_max_distance,
        fold_match_bootstrap_samples=args.fold_match_bootstrap_samples,
        fold_match_bootstrap_seed=args.fold_match_bootstrap_seed,
    )


if __name__ == "__main__":
    main()
