import pandas as pd

from src.backtesting.walk_forward import run_walk_forward
from src.labeling.audit import build_label_summary
from src.labeling.future_return import (
    FutureReturnLabelConfig,
    build_observation_transactions,
    compute_future_return_labels,
)


def test_future_return_labels_use_observation_then_future_window():
    trades = pd.DataFrame([
        {
            "address": "a",
            "market_id": "m0",
            "side": "BUY",
            "amount": 10,
            "price": 0.6,
            "timestamp": "2026-01-01T00:00:00Z",
        },
        {
            "address": "a",
            "market_id": "m1",
            "side": "BUY",
            "amount": 10,
            "price": 0.7,
            "timestamp": "2026-01-03T00:00:00Z",
        },
        {
            "address": "b",
            "market_id": "m2",
            "side": "BUY",
            "amount": 10,
            "price": 0.8,
            "timestamp": "2026-01-01T00:00:00Z",
        },
        {
            "address": "b",
            "market_id": "m3",
            "side": "BUY",
            "amount": 10,
            "price": 0.8,
            "timestamp": "2026-01-03T00:00:00Z",
        },
    ])

    observed = build_observation_transactions(trades, observation_days=1)
    assert set(observed["market_id"]) == {"m0", "m2"}

    labels = compute_future_return_labels(
        trades,
        resolutions={"m1": "Yes", "m3": "No"},
        config=FutureReturnLabelConfig(observation_days=1, horizon_days=1),
    ).set_index("address")

    assert labels.loc["a", "future_net_profit"] > 0
    assert labels.loc["b", "future_net_profit"] < 0
    assert labels.loc["a", "label_source"] == "future_return_pending_threshold"


def test_future_return_label_summary_is_independent():
    df = pd.DataFrame({
        "address": ["a", "b"],
        "split": ["train", "test"],
        "Trader_Success_Rate": [1, 0],
        "label_source": ["future_return", "future_return"],
        "future_n_resolved_trades": [2, 1],
        "future_return_threshold": [0.1, 0.1],
    })

    summary = build_label_summary(df, label_strategy="future_return", requested_label_mode="future_return")

    assert summary["label_independence"] == "independent_future_return"
    assert summary["future_label_coverage"]["eligible_rows"] == 2


def test_walk_forward_accepts_future_return_as_formal_label(monkeypatch, tmp_path):
    import src.backtesting.walk_forward as wf

    class DummyBacktester:
        assumptions = FutureReturnLabelConfig()

        def __init__(self, **kwargs):
            pass

        def load_data(self):
            self.tx_df = pd.DataFrame({
                "address": ["a", "b"],
                "market_id": ["m1", "m2"],
                "side": ["BUY", "BUY"],
                "amount": [10, 10],
                "price": [0.5, 0.5],
                "timestamp": ["2026-01-01", "2026-01-03"],
            })

        def simulate_for_addresses(self, **kwargs):
            return {
                "status": "no_valid_trades",
                "Total_Trades": 0,
                "Win_Rate": 0.0,
                "Total_Net_Profit": 0.0,
                "ROI_Pct": 0.0,
                "Max_Drawdown": 0.0,
                "Sharpe_Ratio (Annualized)": 0.0,
            }

    feature_path = tmp_path / "data" / "features" / "model_input.csv"
    feature_path.parent.mkdir(parents=True)
    pd.DataFrame({
        "address": ["a"],
        "first_trade_date": ["2025-12-31"],
        "Trader_Success_Rate": [1],
        "future_roi": [0.2],
    }).to_csv(feature_path, index=False)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "label_summary.json").write_text(
        '{"label_independence":"independent_future_return","label_strategy":"future_return"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(wf, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(wf, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(wf, "StrategyBacktester", DummyBacktester)

    summary = run_walk_forward(n_folds=1, max_targets=1)

    assert summary["temporal_guard"]["formal_validity"] is True
