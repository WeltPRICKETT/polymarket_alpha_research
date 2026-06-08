from pathlib import Path

import pandas as pd


def test_event_ablation_specs_include_live_baseline_and_target_features():
    from scripts.run_event_ablation import EVENT_ABLATION_SPECS, PAPER_CANDIDATES, RICH_CANDIDATES

    specs = {spec.name: spec.extra_features for spec in EVENT_ABLATION_SPECS}

    assert specs["live"] == []
    assert specs["live_plus_unique_events"] == ["unique_events"]
    assert specs["live_plus_event_notional_hhi_top_share"] == [
        "event_notional_hhi",
        "top_event_notional_share",
    ]
    assert PAPER_CANDIDATES == [
        "live",
        "live_plus_event_diversification",
        "live_plus_mean_markets_per_event_traded",
    ]
    assert "live_plus_topic_diversification" in RICH_CANDIDATES
    assert specs["live_plus_neg_risk_exposure"] == [
        "neg_risk_trade_share",
        "neg_risk_notional_share",
    ]
    assert specs["live_plus_static_event_shape"] == [
        "avg_event_market_count",
        "avg_event_duration_days",
        "avg_creation_to_start_days",
        "neg_risk_trade_share",
        "order_book_trade_share",
    ]


def test_register_ablation_feature_set_keeps_live_features_first():
    from scripts.run_event_ablation import register_ablation_feature_set
    from src.models.feature_sets import EVENT_LIVE_FEATURE_COLS, LIVE_FEATURE_COLS, get_feature_columns

    feature_set = register_ablation_feature_set(
        "live_plus_unique_events",
        ["unique_events"],
    )

    assert get_feature_columns(feature_set) == LIVE_FEATURE_COLS + ["unique_events"]
    assert "total_roi" not in get_feature_columns(feature_set)
    assert set(["unique_events"]).issubset(EVENT_LIVE_FEATURE_COLS)


def test_write_outputs_creates_academic_ablation_table_and_report(tmp_path):
    from scripts.run_event_ablation import AblationResult, write_outputs

    results = [
        AblationResult(
            name="live",
            feature_set="event_ablation__live",
            extra_features=[],
            run_dir=tmp_path / "live",
            metrics={
                "direct_split_auc": 0.61,
                "direct_split_average_precision": 0.28,
                "rolling_auc": 0.54,
                "rolling_average_precision": 0.23,
                "precision_at_20": 0.2,
                "top_10_percent_precision": 0.23,
                "walk_forward_total_net_profit": 100.0,
                "walk_forward_mean_equal_capital_roi": 1.5,
                "model_vs_random_percentile": 0.8,
                "bootstrap_prob_beats_matched_random": 0.7,
                "drop_top_1_profit_event_remaining_net_profit": 50.0,
                "drop_top_3_profit_events_remaining_net_profit": 10.0,
                "top_event_profit_share": 0.5,
            },
        )
    ]

    table_path, report_path = write_outputs(
        results,
        output_dir=tmp_path,
        run_name="event_ablation_smoke",
    )

    table = pd.read_csv(table_path)
    assert table_path.name == "event_ablation_smoke_table.csv"
    assert report_path.name == "event_ablation_smoke_report.md"
    assert table.loc[0, "ablation"] == "live"
    for column in [
        "direct_split_auc",
        "rolling_average_precision",
        "walk_forward_total_net_profit",
        "bootstrap_prob_beats_matched_random",
        "drop_top_1_profit_event_remaining_net_profit",
        "top_event_profit_share",
    ]:
        assert column in table.columns
    assert "event ablation" in Path(report_path).read_text(encoding="utf-8").lower()


def test_paper_runner_uses_safe_candidate_matrix_without_credentials():
    script = Path("scripts/run_event_ablation_paper.sh")

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "--paper-candidates" in text
    assert "--run-name event_ablation_paper" in text
    assert "FOLD_MATCH_BOOTSTRAP_SAMPLES" in text
    assert "BASELINE_SEED_COUNT" in text
    sensitive_password = "88" * 4
    sensitive_host = ".".join(["10", "33", "104", "34"])
    assert sensitive_password not in text
    assert sensitive_host not in text
