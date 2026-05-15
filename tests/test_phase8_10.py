import pandas as pd

from src.experiments.future_label_experiments import (
    build_future_label_frame,
    make_rolling_origin_folds,
    run_model_informed_walk_forward,
    run_exposure_concentration_analysis,
    run_fold_matched_random_baseline_analysis,
    run_random_baseline_stability,
    run_stratified_random_baseline_analysis,
    run_label_sensitivity,
    run_rolling_origin_evaluation,
)
from src.labeling.future_return import FutureReturnLabelConfig


def _toy_trades():
    rows = []
    for i in range(12):
        address = f"a{i}"
        rows.append({
            "address": address,
            "market_id": f"obs{i}",
            "side": "BUY",
            "amount": 10,
            "price": 0.5,
            "timestamp": f"2026-01-{i + 1:02d}T00:00:00Z",
        })
        rows.append({
            "address": address,
            "market_id": f"future{i}",
            "side": "BUY",
            "amount": 10,
            "price": 0.8 if i % 3 == 0 else 0.2,
            "timestamp": f"2026-01-{i + 2:02d}T00:00:00Z",
        })
    return pd.DataFrame(rows)


def test_future_label_frame_splits_after_threshold():
    trades = _toy_trades()
    resolutions = {f"future{i}": "Yes" if i % 3 == 0 else "No" for i in range(12)}
    labeled = build_future_label_frame(
        trades,
        resolutions,
        FutureReturnLabelConfig(observation_days=0, horizon_days=2, top_percentile=0.25),
        train_ratio=0.5,
        val_ratio=0.25,
    )

    assert set(labeled["label_source"]) == {"future_return"}
    assert {"train", "val", "test"}.issubset(set(labeled["split"]))
    assert "future_return_threshold" in labeled.columns


def test_label_sensitivity_writes_recommendation(tmp_path, monkeypatch):
    import src.experiments.future_label_experiments as exp

    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    report = run_label_sensitivity(
        observation_days=[0],
        horizon_days=[2],
        min_trades_values=[1],
        top_percentiles=[0.25],
        min_test_rows=1,
        min_test_positive=1,
        trades_df=_toy_trades(),
        resolutions={f"future{i}": "Yes" if i % 3 == 0 else "No" for i in range(12)},
    )

    assert report["status"] == "ok"
    assert (tmp_path / "future_label_sensitivity.csv").exists()
    assert "recommended_config" in report


def test_rolling_origin_evaluation_has_valid_fold(tmp_path, monkeypatch):
    import src.experiments.future_label_experiments as exp

    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    feature_path = tmp_path / "model_input.csv"
    df = pd.DataFrame({
        "address": [f"a{i}" for i in range(20)],
        "first_trade_date": pd.date_range("2026-01-01", periods=20, freq="D"),
        "Trader_Success_Rate": [0, 1] * 10,
        "early_entry_score": [i / 20 for i in range(20)],
        "contrarian_score": [0.2] * 20,
        "information_ratio": [0.1] * 20,
        "cross_market_diversification": [0.3] * 20,
        "avg_holding_period": [1.0] * 20,
        "trading_frequency": [2.0] * 20,
        "capital_flow_centrality": [0.01] * 20,
    })
    df.to_csv(feature_path, index=False)

    folds = make_rolling_origin_folds(df, n_folds=3, min_train_fraction=0.4)
    summary = run_rolling_origin_evaluation(
        feature_path=feature_path,
        n_folds=3,
        min_train_fraction=0.4,
        top_k=3,
    )

    assert len(folds) == 3
    assert summary["valid_fold_count"] >= 1
    assert (tmp_path / "rolling_origin_results.csv").exists()
    assert (tmp_path / "rolling_origin_predictions.csv").exists()
    assert (tmp_path / "threshold_policy_results.csv").exists()
    assert "top_k_10" in summary["threshold_policies"]


def test_model_informed_walk_forward_uses_prediction_targets(tmp_path, monkeypatch):
    import src.experiments.future_label_experiments as exp

    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    predictions = pd.DataFrame({
        "fold": [1, 1, 1],
        "address": ["a", "b", "c"],
        "predicted_probability": [0.9, 0.2, 0.1],
        "test_start": ["2026-01-02", "2026-01-02", "2026-01-02"],
        "test_end": ["2026-01-05", "2026-01-05", "2026-01-05"],
    })
    pred_path = tmp_path / "rolling_origin_predictions.csv"
    predictions.to_csv(pred_path, index=False)

    class DummyBacktester:
        assumptions = type("A", (), {"__dict__": {}})()

        def __init__(self, **kwargs):
            self.tx_df = pd.DataFrame()

        def load_data(self):
            self.tx_df = pd.DataFrame({
                "address": ["a", "b", "c"],
                "market_id": ["m1", "m1", "m1"],
                "side": ["BUY", "BUY", "BUY"],
                "amount": [10, 10, 10],
                "price": [0.5, 0.5, 0.5],
                "timestamp": ["2026-01-01", "2026-01-01", "2026-01-01"],
            })

        def simulate_for_addresses(self, target_addresses, strategy_name, **kwargs):
            return {
                "status": "ok",
                "Total_Trades": len(target_addresses),
                "Win_Rate": 0.5,
                "Total_Net_Profit": float(len(target_addresses)),
                "Total_Filled_Notional": float(len(target_addresses)),
                "Fill_Rate": 1.0,
                "ROI_Pct": 0.1,
                "Max_Drawdown": 0.0,
                "Sharpe_Ratio (Annualized)": 1.0,
            }

    monkeypatch.setattr(exp, "StrategyBacktester", DummyBacktester)

    summary = run_model_informed_walk_forward(predictions_path=pred_path, top_k=1, top_percentile=0.5)

    assert summary["status"] == "ok"
    assert summary["aggregate"]["model_top_k_1"]["total_trades"] == 1
    assert "mean_equal_capital_roi" in summary["aggregate"]["model_top_k_1"]
    assert (tmp_path / "model_informed_walk_forward_results.csv").exists()


def test_random_baseline_stability_compares_model_summary(tmp_path, monkeypatch):
    import json
    import src.experiments.future_label_experiments as exp

    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    predictions = pd.DataFrame({
        "fold": [1, 1, 1, 1],
        "address": ["a", "b", "c", "d"],
        "predicted_probability": [0.9, 0.7, 0.2, 0.1],
        "test_start": ["2026-01-02"] * 4,
        "test_end": ["2026-01-05"] * 4,
    })
    pred_path = tmp_path / "rolling_origin_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    model_summary_path = tmp_path / "model_informed_walk_forward_summary.json"
    model_summary_path.write_text(json.dumps({
        "aggregate": {
            "model_top_pct_0.10": {
                "total_net_profit": 5.0,
                "mean_equal_capital_roi": 1.0,
                "total_equal_capital_net_profit": 5.0,
            }
        }
    }), encoding="utf-8")

    class DummyBacktester:
        def __init__(self, **kwargs):
            self.tx_df = pd.DataFrame()

        def load_data(self):
            self.tx_df = pd.DataFrame({
                "address": ["a", "b", "c", "d"],
                "market_id": ["m1"] * 4,
                "side": ["BUY"] * 4,
                "amount": [10, 10, 10, 10],
                "price": [0.5, 0.5, 0.5, 0.5],
                "timestamp": ["2026-01-01"] * 4,
            })

        def simulate_for_addresses(self, target_addresses, strategy_name, **kwargs):
            n = len(target_addresses)
            return {
                "status": "ok",
                "Total_Trades": n,
                "Requested_Trades": n,
                "Total_Net_Profit": float(n),
                "Total_Filled_Notional": float(n),
                "Fill_Rate": 1.0,
                "ROI_Pct": 1.0,
                "Max_Drawdown": 0.0,
                "Sharpe_Ratio (Annualized)": 1.0,
            }

    monkeypatch.setattr(exp, "StrategyBacktester", DummyBacktester)

    summary = run_random_baseline_stability(
        predictions_path=pred_path,
        model_summary_path=model_summary_path,
        seed_count=3,
        top_k=2,
    )

    assert summary["status"] == "ok"
    assert summary["distributions"]["total_net_profit"]["count"] == 3
    assert summary["model_vs_random"]["total_net_profit"]["status"] == "ok"
    assert (tmp_path / "random_baseline_stability.csv").exists()
    assert (tmp_path / "random_baseline_seed_summary.csv").exists()


def test_exposure_concentration_analysis_compares_random_seeds(tmp_path, monkeypatch):
    import json
    import src.experiments.future_label_experiments as exp

    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    model_summary_path = tmp_path / "model_informed_walk_forward_summary.json"
    model_summary_path.write_text(json.dumps({
        "aggregate": {
            "model_top_pct_0.10": {
                "mean_unique_markets": 3.0,
                "mean_top_market_share": 0.5,
                "mean_market_hhi": 0.4,
                "mean_effective_markets": 2.5,
            }
        }
    }), encoding="utf-8")
    seed_summary_path = tmp_path / "random_baseline_seed_summary.csv"
    pd.DataFrame([
        {
            "seed_index": 0,
            "mean_unique_markets": 2.0,
            "mean_top_market_share": 0.8,
            "mean_market_hhi": 0.7,
            "mean_effective_markets": 1.4,
        },
        {
            "seed_index": 1,
            "mean_unique_markets": 4.0,
            "mean_top_market_share": 0.4,
            "mean_market_hhi": 0.3,
            "mean_effective_markets": 3.3,
        },
    ]).to_csv(seed_summary_path, index=False)

    summary = run_exposure_concentration_analysis(
        model_summary_path=model_summary_path,
        random_seed_summary_path=seed_summary_path,
    )

    assert summary["status"] == "ok"
    assert summary["model_vs_random"]["mean_market_hhi"]["status"] == "ok"
    assert "left_tail_p_value" in summary["model_vs_random"]["mean_market_hhi"]
    assert (tmp_path / "exposure_concentration_summary.json").exists()


def test_stratified_random_baseline_matches_execution_profile(tmp_path, monkeypatch):
    import json
    import src.experiments.future_label_experiments as exp

    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    model_summary_path = tmp_path / "model_informed_walk_forward_summary.json"
    model_summary_path.write_text(json.dumps({
        "aggregate": {
            "model_top_pct_0.10": {
                "total_trades": 100,
                "mean_fill_rate": 0.50,
                "mean_market_hhi": 0.10,
                "mean_top_market_share": 0.20,
                "mean_effective_markets": 10.0,
                "total_net_profit": 20.0,
                "mean_equal_capital_roi": 2.0,
                "total_equal_capital_net_profit": 10.0,
            }
        }
    }), encoding="utf-8")
    seed_summary_path = tmp_path / "random_baseline_seed_summary.csv"
    pd.DataFrame([
        {
            "seed_index": 0,
            "total_trades": 98,
            "mean_fill_rate": 0.51,
            "mean_market_hhi": 0.11,
            "mean_top_market_share": 0.19,
            "mean_effective_markets": 9.5,
            "total_net_profit": 5.0,
            "mean_equal_capital_roi": 0.5,
            "total_equal_capital_net_profit": 3.0,
        },
        {
            "seed_index": 1,
            "total_trades": 300,
            "mean_fill_rate": 0.90,
            "mean_market_hhi": 0.70,
            "mean_top_market_share": 0.80,
            "mean_effective_markets": 1.5,
            "total_net_profit": 50.0,
            "mean_equal_capital_roi": 5.0,
            "total_equal_capital_net_profit": 30.0,
        },
        {
            "seed_index": 2,
            "total_trades": 105,
            "mean_fill_rate": 0.49,
            "mean_market_hhi": 0.09,
            "mean_top_market_share": 0.21,
            "mean_effective_markets": 10.5,
            "total_net_profit": 6.0,
            "mean_equal_capital_roi": 0.6,
            "total_equal_capital_net_profit": 4.0,
        },
    ]).to_csv(seed_summary_path, index=False)

    summary = run_stratified_random_baseline_analysis(
        model_summary_path=model_summary_path,
        random_seed_summary_path=seed_summary_path,
        match_count=2,
    )

    assert summary["status"] == "ok"
    assert summary["matched_seed_count"] == 2
    assert summary["matched_seed_indices"] == [0, 2]
    assert summary["model_vs_matched_random"]["total_net_profit"]["status"] == "ok"
    assert (tmp_path / "stratified_random_baseline_matches.csv").exists()
    assert (tmp_path / "stratified_random_baseline_summary.json").exists()


def test_fold_matched_random_baseline_matches_within_each_fold(tmp_path, monkeypatch):
    import src.experiments.future_label_experiments as exp

    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    model_path = tmp_path / "model_informed_walk_forward_results.csv"
    pd.DataFrame([
        {
            "fold": 1,
            "strategy": "model_top_pct_0.10",
            "total_trades": 10,
            "requested_trades": 12,
            "total_filled_notional": 100.0,
            "fill_rate": 0.50,
            "total_net_profit": 10.0,
            "roi_pct": 0.1,
            "market_hhi": 0.20,
            "top_market_share": 0.30,
            "effective_markets": 5.0,
            "equal_capital_net_profit": 8.0,
            "equal_capital_roi": 0.8,
        },
        {
            "fold": 2,
            "strategy": "model_top_pct_0.10",
            "total_trades": 20,
            "requested_trades": 25,
            "total_filled_notional": 200.0,
            "fill_rate": 0.40,
            "total_net_profit": 20.0,
            "roi_pct": 0.2,
            "market_hhi": 0.10,
            "top_market_share": 0.20,
            "effective_markets": 10.0,
            "equal_capital_net_profit": 12.0,
            "equal_capital_roi": 1.2,
        },
    ]).to_csv(model_path, index=False)
    random_path = tmp_path / "random_baseline_stability.csv"
    pd.DataFrame([
        {
            "fold": 1,
            "seed_index": 0,
            "strategy": "baseline_random_wallets",
            "total_trades": 11,
            "requested_trades": 12,
            "total_filled_notional": 90.0,
            "fill_rate": 0.49,
            "total_net_profit": 2.0,
            "roi_pct": 0.02,
            "market_hhi": 0.21,
            "top_market_share": 0.29,
            "effective_markets": 5.2,
            "equal_capital_net_profit": 1.0,
            "equal_capital_roi": 0.1,
        },
        {
            "fold": 1,
            "seed_index": 1,
            "strategy": "baseline_random_wallets",
            "total_trades": 80,
            "requested_trades": 90,
            "total_filled_notional": 900.0,
            "fill_rate": 0.95,
            "total_net_profit": 50.0,
            "roi_pct": 0.5,
            "market_hhi": 0.80,
            "top_market_share": 0.90,
            "effective_markets": 1.2,
            "equal_capital_net_profit": 30.0,
            "equal_capital_roi": 3.0,
        },
        {
            "fold": 2,
            "seed_index": 0,
            "strategy": "baseline_random_wallets",
            "total_trades": 19,
            "requested_trades": 25,
            "total_filled_notional": 210.0,
            "fill_rate": 0.41,
            "total_net_profit": 3.0,
            "roi_pct": 0.03,
            "market_hhi": 0.11,
            "top_market_share": 0.19,
            "effective_markets": 9.8,
            "equal_capital_net_profit": 2.0,
            "equal_capital_roi": 0.2,
        },
        {
            "fold": 2,
            "seed_index": 1,
            "strategy": "baseline_random_wallets",
            "total_trades": 5,
            "requested_trades": 6,
            "total_filled_notional": 30.0,
            "fill_rate": 0.10,
            "total_net_profit": -5.0,
            "roi_pct": -0.1,
            "market_hhi": 0.50,
            "top_market_share": 0.60,
            "effective_markets": 2.0,
            "equal_capital_net_profit": -3.0,
            "equal_capital_roi": -0.3,
        },
    ]).to_csv(random_path, index=False)

    summary = run_fold_matched_random_baseline_analysis(
        model_results_path=model_path,
        random_results_path=random_path,
        match_count_per_fold=2,
        max_match_distance=0.2,
        bootstrap_samples=25,
    )

    assert summary["status"] == "ok"
    assert summary["matched_rank_count"] == 1
    assert summary["matched_rows"] == 2
    assert summary["caliper_passed_folds"] == 2
    assert summary["model_vs_fold_matched_random"]["total_net_profit"]["status"] == "ok"
    assert summary["bootstrap_model_minus_matched_random"]["status"] == "ok"
    assert (
        summary["bootstrap_model_minus_matched_random"]["intervals"]["total_net_profit"]
        ["probability_model_beats_matched_random"] == 1.0
    )
    matches = pd.read_csv(tmp_path / "fold_matched_random_baseline_matches.csv")
    assert matches["seed_index"].tolist() == [0, 0]
    assert (tmp_path / "fold_matched_random_baseline_rank_summary.csv").exists()
    assert (tmp_path / "fold_matched_random_bootstrap.csv").exists()
    assert (tmp_path / "fold_matched_random_baseline_summary.json").exists()
