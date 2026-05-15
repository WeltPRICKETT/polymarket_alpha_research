import pandas as pd


def test_backtest_metrics_include_assumptions_for_valid_trade():
    from src.backtesting.engine import StrategyBacktester

    bt = StrategyBacktester(latency_minutes=5, trade_size=100.0, fees_pct=0.001)
    bt.tx_df = pd.DataFrame([
        {
            "transaction_id": "tx1",
            "address": "expert",
            "market_id": "m1",
            "side": "BUY",
            "amount": 10.0,
            "price": 0.4,
            "volume": 1000.0,
            "timestamp": "2026-01-01T00:00:00Z",
        },
        {
            "transaction_id": "tx2",
            "address": "other",
            "market_id": "m1",
            "side": "BUY",
            "amount": 1.0,
            "price": 0.45,
            "volume": 1000.0,
            "timestamp": "2026-01-01T00:05:00Z",
        },
    ])
    bt.tx_df["timestamp"] = pd.to_datetime(bt.tx_df["timestamp"])
    bt.resolutions = {"m1": "NO"}

    metrics = bt.simulate_for_addresses(
        target_addresses=["expert"],
        strategy_name="unit_test_strategy",
        output_prefix=None,
    )

    assert metrics["status"] == "ok"
    assert metrics["Total_Trades"] == 1
    assert metrics["Strategy"] == "unit_test_strategy"
    assert metrics["assumptions"]["latency_minutes"] == 5
    assert metrics["assumptions"]["liquidity_model"] == "capped_by_recent_market_volume_and_participation_rate"
    assert metrics["Total_Filled_Notional"] == 100.0
    assert metrics["Fill_Rate"] > 0.99


def test_backtest_applies_capital_and_liquidity_constraints():
    from src.backtesting.engine import StrategyBacktester

    bt = StrategyBacktester(
        latency_minutes=5,
        trade_size=1000.0,
        fees_pct=0.0,
        max_trade_fraction_of_balance=0.01,
        max_market_exposure_fraction=0.50,
        max_participation_rate=0.10,
        min_fill_notional=1.0,
        price_impact_bps=0.0,
    )
    bt.tx_df = pd.DataFrame([
        {
            "transaction_id": "tx1",
            "address": "expert",
            "market_id": "m1",
            "side": "BUY",
            "amount": 10.0,
            "price": 0.4,
            "volume": 10000.0,
            "timestamp": "2026-01-01T00:00:00Z",
        },
        {
            "transaction_id": "tx2",
            "address": "other",
            "market_id": "m1",
            "side": "BUY",
            "amount": 1.0,
            "price": 0.45,
            "volume": 10000.0,
            "timestamp": "2026-01-01T00:05:00Z",
        },
    ])
    bt.tx_df["timestamp"] = pd.to_datetime(bt.tx_df["timestamp"])
    bt.resolutions = {"m1": "NO"}

    metrics = bt.simulate_for_addresses(
        target_addresses=["expert"],
        strategy_name="constraint_test",
        output_prefix=None,
    )

    assert metrics["status"] == "ok"
    assert metrics["Requested_Trades"] == 1
    assert metrics["Total_Filled_Notional"] == 100.0
    assert abs(metrics["Fill_Rate"] - 0.1) < 1e-5
    assert metrics["Capital_Capped_Trades"] == 1


def test_backtest_no_trade_result_is_auditable():
    from src.backtesting.engine import StrategyBacktester

    bt = StrategyBacktester()
    bt.tx_df = pd.DataFrame(columns=["address", "market_id", "side", "amount", "price", "timestamp"])
    bt.resolutions = {}

    metrics = bt.simulate_for_addresses(
        target_addresses=[],
        strategy_name="empty_strategy",
        output_prefix=None,
    )

    assert metrics["status"] == "no_valid_trades"
    assert metrics["Total_Trades"] == 0
    assert "assumptions" in metrics
    assert "reason" in metrics


def test_backtest_no_fill_result_keeps_execution_diagnostics():
    from src.backtesting.engine import StrategyBacktester

    bt = StrategyBacktester(
        trade_size=100.0,
        max_participation_rate=0.001,
        min_fill_notional=10.0,
    )
    bt.tx_df = pd.DataFrame([
        {
            "transaction_id": "tx1",
            "address": "expert",
            "market_id": "m1",
            "side": "BUY",
            "amount": 10.0,
            "price": 0.4,
            "volume": 100.0,
            "timestamp": "2026-01-01T00:00:00Z",
        },
    ])
    bt.tx_df["timestamp"] = pd.to_datetime(bt.tx_df["timestamp"])
    bt.resolutions = {"m1": "NO"}

    metrics = bt.simulate_for_addresses(
        target_addresses=["expert"],
        strategy_name="no_fill_strategy",
        output_prefix=None,
    )

    assert metrics["status"] == "no_valid_trades"
    assert metrics["Requested_Trades"] == 1
    assert metrics["Skipped_Liquidity"] == 1
    assert metrics["Fill_Rate"] == 0.0


def test_market_concentration_metrics_use_filled_notional_shares():
    from src.backtesting.engine import market_concentration_metrics

    ledger = pd.DataFrame([
        {"market_id": "m1", "filled_notional": 30.0},
        {"market_id": "m1", "filled_notional": 30.0},
        {"market_id": "m2", "filled_notional": 40.0},
    ])

    metrics = market_concentration_metrics(ledger)

    assert metrics["Unique_Markets"] == 2
    assert abs(metrics["Top_Market_Filled_Notional_Share"] - 0.6) < 1e-9
    assert abs(metrics["Market_HHI"] - 0.52) < 1e-9
    assert abs(metrics["Effective_Markets"] - (1 / 0.52)) < 1e-9


def test_equal_capital_metrics_use_fold_common_budget():
    from src.backtesting.equal_capital import add_equal_capital_metrics

    rows = pd.DataFrame([
        {"fold": 1, "strategy": "a", "total_filled_notional": 100.0, "total_net_profit": 10.0},
        {"fold": 1, "strategy": "b", "total_filled_notional": 50.0, "total_net_profit": 10.0},
        {"fold": 2, "strategy": "a", "total_filled_notional": 0.0, "total_net_profit": 0.0},
        {"fold": 2, "strategy": "b", "total_filled_notional": 20.0, "total_net_profit": -10.0},
    ])

    enriched = add_equal_capital_metrics(rows)
    fold1 = enriched[enriched["fold"] == 1].set_index("strategy")
    fold2 = enriched[enriched["fold"] == 2].set_index("strategy")

    assert fold1.loc["a", "equal_capital_budget"] == 50.0
    assert fold1.loc["a", "equal_capital_net_profit"] == 5.0
    assert fold1.loc["b", "equal_capital_net_profit"] == 10.0
    assert not bool(fold2.loc["a", "equal_capital_supported"])
    assert fold2.loc["b", "equal_capital_budget"] == 20.0


def test_walk_forward_target_selection_is_chronological():
    from src.backtesting.walk_forward import _select_label_targets

    features = pd.DataFrame([
        {"address": "future", "first_trade_date": "2026-01-03", "Trader_Success_Rate": 1, "composite_score": 0.99},
        {"address": "past", "first_trade_date": "2026-01-01", "Trader_Success_Rate": 1, "composite_score": 0.50},
        {"address": "noise", "first_trade_date": "2026-01-01", "Trader_Success_Rate": 0, "composite_score": 1.00},
    ])

    targets = _select_label_targets(features, pd.Timestamp("2026-01-02"), max_targets=10)

    assert targets == ["past"]


def test_api_exposes_walk_forward_results_route():
    from src.visualization.api import app

    paths = {route.path for route in app.routes}
    assert "/api/backtest/walk-forward" in paths
