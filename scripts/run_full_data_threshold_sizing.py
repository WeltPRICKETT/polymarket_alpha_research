#!/usr/bin/env python
"""Threshold and sizing diagnostics for full-data Step 1 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLDS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _profit_concentration(selected: pd.DataFrame) -> dict[str, float]:
    profits = selected.groupby("address")["future_net_profit"].sum().sort_values(ascending=False)
    total = float(profits.sum()) if len(profits) else 0.0
    positive_total = float(profits[profits > 0].sum()) if len(profits) else 0.0
    top1 = float(profits.iloc[0]) if len(profits) else 0.0
    top3 = float(profits.head(3).sum()) if len(profits) else 0.0
    top10 = float(profits.head(10).sum()) if len(profits) else 0.0
    return {
        "top_1_wallet_profit": top1,
        "top_3_wallet_profit": top3,
        "top_10_wallet_profit": top10,
        "profit_after_drop_top_1_wallet": total - top1,
        "profit_after_drop_top_3_wallets": total - top3,
        "top_1_wallet_positive_profit_share": top1 / positive_total if positive_total else np.nan,
        "top_3_wallet_positive_profit_share": top3 / positive_total if positive_total else np.nan,
        "top_10_wallet_positive_profit_share": top10 / positive_total if positive_total else np.nan,
    }


def evaluate_thresholds(pred: pd.DataFrame, random_seeds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    rows = []
    fold_rows = []
    pred = pred.replace([np.inf, -np.inf], np.nan).fillna(0)
    pred["future_invested"] = pred["future_invested"].clip(lower=0)

    grouped = {
        (regime, int(fold)): group.sort_values("score", ascending=False).reset_index(drop=True)
        for (regime, fold), group in pred.groupby(["regime", "fold"], sort=True)
    }

    for regime in sorted(pred["regime"].unique()):
        regime_groups = {fold: group for (reg, fold), group in grouped.items() if reg == regime}
        for threshold in THRESHOLDS:
            selected_parts = []
            for fold, group in regime_groups.items():
                k = max(1, int(np.ceil(len(group) * threshold)))
                selected = group.head(k).copy()
                selected_parts.append(selected)
                fold_rows.append(
                    {
                        "regime": regime,
                        "threshold": threshold,
                        "fold": fold,
                        "test_wallets": int(len(group)),
                        "selected_wallets": int(len(selected)),
                        "precision": float(selected["Trader_Success_Rate"].mean()),
                        "gross_future_net_profit": float(selected["future_net_profit"].sum()),
                        "actual_capital_roi": float(selected["future_net_profit"].sum() / selected["future_invested"].sum())
                        if selected["future_invested"].sum() > 0
                        else np.nan,
                        "equal_wallet_mean_roi": float(selected["future_roi"].mean()),
                        "score_weighted_mean_roi": float(np.average(selected["future_roi"], weights=selected["score"].clip(lower=0)))
                        if selected["score"].clip(lower=0).sum() > 0
                        else np.nan,
                    }
                )
            all_selected = pd.concat(selected_parts, ignore_index=True)
            total_profit = float(all_selected["future_net_profit"].sum())
            total_invested = float(all_selected["future_invested"].sum())
            equal_roi = float(all_selected["future_roi"].mean())
            score_weights = all_selected["score"].clip(lower=0)
            score_roi = float(np.average(all_selected["future_roi"], weights=score_weights)) if score_weights.sum() > 0 else np.nan
            random_profit = []
            random_equal_roi = []
            random_actual_roi = []
            for _ in range(random_seeds):
                parts = []
                for fold, group in regime_groups.items():
                    k = max(1, int(np.ceil(len(group) * threshold)))
                    idx = rng.choice(len(group), size=min(k, len(group)), replace=False)
                    parts.append(group.iloc[idx])
                picks = pd.concat(parts, ignore_index=True)
                random_profit.append(float(picks["future_net_profit"].sum()))
                random_equal_roi.append(float(picks["future_roi"].mean()))
                invested = float(picks["future_invested"].sum())
                random_actual_roi.append(float(picks["future_net_profit"].sum() / invested) if invested > 0 else np.nan)
            conc = _profit_concentration(all_selected)
            rows.append(
                {
                    "regime": regime,
                    "threshold": threshold,
                    "selected_wallet_rows": int(len(all_selected)),
                    "unique_wallets": int(all_selected["address"].nunique()),
                    "precision": float(all_selected["Trader_Success_Rate"].mean()),
                    "gross_future_net_profit": total_profit,
                    "actual_capital_roi": total_profit / total_invested if total_invested > 0 else np.nan,
                    "equal_wallet_mean_roi": equal_roi,
                    "score_weighted_mean_roi": score_roi,
                    "positive_profit_folds": int(
                        sum(
                            1
                            for part in selected_parts
                            if float(part["future_net_profit"].sum()) > 0
                        )
                    ),
                    "positive_equal_roi_folds": int(
                        sum(
                            1
                            for part in selected_parts
                            if float(part["future_roi"].mean()) > 0
                        )
                    ),
                    "random_profit_percentile": float((np.array(random_profit) < total_profit).mean()),
                    "random_equal_roi_percentile": float((np.array(random_equal_roi) < equal_roi).mean()),
                    "random_actual_roi_percentile": float((np.array(random_actual_roi) < (total_profit / total_invested if total_invested > 0 else np.nan)).mean()),
                    "random_profit_p025": float(np.nanquantile(random_profit, 0.025)),
                    "random_profit_p975": float(np.nanquantile(random_profit, 0.975)),
                    "random_equal_roi_p025": float(np.nanquantile(random_equal_roi, 0.025)),
                    "random_equal_roi_p975": float(np.nanquantile(random_equal_roi, 0.975)),
                    **conc,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def render_markdown(summary: pd.DataFrame, fold_summary: pd.DataFrame) -> str:
    def table(df: pd.DataFrame) -> str:
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
        return "\n".join(lines)

    best_profit = summary.sort_values("gross_future_net_profit", ascending=False).head(10)
    best_equal_roi = summary.sort_values("equal_wallet_mean_roi", ascending=False).head(10)
    best_precision = summary.sort_values("precision", ascending=False).head(10)
    positive_roi = summary[summary["equal_wallet_mean_roi"] > 0].copy()

    lines = [
        "# Full-Data Threshold and Sizing Sweep",
        "",
        "Generated: 2026-05-22",
        "",
        "This diagnostic uses out-of-sample rolling predictions from the true-live full-data Step 1 run.",
        "",
        "## Best Gross Profit",
        "",
        table(best_profit[["regime", "threshold", "selected_wallet_rows", "precision", "gross_future_net_profit", "actual_capital_roi", "equal_wallet_mean_roi", "random_profit_percentile", "positive_profit_folds", "profit_after_drop_top_1_wallet", "profit_after_drop_top_3_wallets"]]),
        "",
        "## Best Equal-Wallet ROI",
        "",
        table(best_equal_roi[["regime", "threshold", "selected_wallet_rows", "precision", "gross_future_net_profit", "actual_capital_roi", "equal_wallet_mean_roi", "random_equal_roi_percentile", "positive_equal_roi_folds"]]),
        "",
        "## Best Precision",
        "",
        table(best_precision[["regime", "threshold", "selected_wallet_rows", "precision", "gross_future_net_profit", "actual_capital_roi", "equal_wallet_mean_roi"]]),
        "",
        "## Positive Equal-Wallet ROI Candidates",
        "",
        table(positive_roi[["regime", "threshold", "selected_wallet_rows", "precision", "gross_future_net_profit", "actual_capital_roi", "equal_wallet_mean_roi", "random_equal_roi_percentile", "positive_equal_roi_folds"]])
        if not positive_roi.empty
        else "No threshold produced positive equal-wallet mean ROI.",
        "",
        "## Interpretation",
        "",
    ]
    if positive_roi.empty:
        lines.append("- Thresholding alone did not convert the true-live wallet ranking into positive equal-wallet ROI.")
    else:
        lines.append("- At least one threshold produced positive equal-wallet ROI; these rows should be promoted to price-impact and execution diagnostics.")
    top = best_profit.iloc[0]
    lines.append(
        f"- Best gross-profit row: `{top['regime']}` at top {top['threshold']:.2%}, gross future net profit {top['gross_future_net_profit']:.2f}, equal-wallet ROI {top['equal_wallet_mean_roi']:.4f}."
    )
    lines.append("- Treat gross profit and equal-wallet ROI separately: gross profit can be dominated by high-notional future wallets, while equal-wallet ROI is closer to a normalized follow-wallet strategy.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="results/new_data_sources/full_data_step1_true_live/full_data_step1_predictions.csv")
    parser.add_argument("--output-dir", default="results/new_data_sources/full_data_step1_true_live")
    parser.add_argument("--random-seeds", type=int, default=200)
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, fold_summary = evaluate_thresholds(pred, args.random_seeds)
    summary.to_csv(out_dir / "full_data_threshold_sizing_summary.csv", index=False)
    fold_summary.to_csv(out_dir / "full_data_threshold_sizing_folds.csv", index=False)
    report = {
        "thresholds": THRESHOLDS,
        "random_seeds": args.random_seeds,
        "prediction_rows": int(len(pred)),
        "summary_rows": int(len(summary)),
    }
    (out_dir / "full_data_threshold_sizing_summary.json").write_text(json.dumps(report, indent=2))
    (out_dir / "full_data_threshold_sizing_report.md").write_text(render_markdown(summary, fold_summary))
    print(summary.sort_values("gross_future_net_profit", ascending=False).head(10).to_string(index=False))
    print(f"Wrote threshold/sizing diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
