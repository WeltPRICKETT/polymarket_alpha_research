import pandas as pd


def test_live_feature_set_excludes_post_resolution_features():
    from src.models.feature_sets import FEATURE_LEAKAGE_CLASS, get_feature_columns

    live_cols = get_feature_columns("live")
    assert "total_roi" not in live_cols
    assert "win_rate" not in live_cols
    assert all(FEATURE_LEAKAGE_CLASS[col] == "pre_resolution_live" for col in live_cols)


def test_live_event_feature_set_adds_pre_resolution_event_context():
    from src.models.feature_sets import FEATURE_LEAKAGE_CLASS, get_feature_columns

    live_event_cols = get_feature_columns("live_event")

    assert "total_roi" not in live_event_cols
    assert "unique_events" in live_event_cols
    assert "event_notional_hhi" in live_event_cols
    assert FEATURE_LEAKAGE_CLASS["unique_events"] == "pre_resolution_event_context"


def test_research_feature_set_keeps_explanatory_features():
    from src.models.feature_sets import get_feature_columns

    research_cols = get_feature_columns("research")
    assert "total_roi" in research_cols
    assert "win_rate" in research_cols
    assert "Risk_Adjusted_Return" not in research_cols
    assert "Risk_Adjusted_Return" in get_feature_columns("research_with_rar")


def test_label_summary_flags_single_class_split():
    from src.labeling.audit import build_label_summary

    df = pd.DataFrame({
        "address": ["a", "b", "c", "d"],
        "split": ["train", "train", "test", "test"],
        "Trader_Success_Rate": [0, 1, 0, 0],
        "label_source": ["resolution", "resolution", "resolution", "resolution"],
    })

    summary = build_label_summary(df, label_strategy="resolution", requested_label_mode="resolution")

    assert summary["label_independence"] == "independent_resolution"
    assert summary["single_class_splits"] == ["test"]
    assert summary["is_trainable_for_binary_eval"] is False


def test_composite_label_audit_marks_basis_features():
    from src.labeling.audit import build_label_audit

    df = pd.DataFrame({
        "address": ["a"],
        "split": ["train"],
        "Trader_Success_Rate": [1],
        "label_source": ["composite_rank"],
        "composite_score": [0.95],
    })

    audit_df = build_label_audit(df, label_strategy="composite", requested_label_mode="auto")

    assert audit_df.loc[0, "label_independence"] == "feature_derived_exploratory"
    assert audit_df.loc[0, "label_basis_features"] == "total_roi,win_rate,profit_loss_ratio"


def test_trainer_defaults_to_live_feature_set():
    from src.models.trainer import ModelTrainer

    trainer = ModelTrainer()

    assert trainer.feature_set == "live"
    assert "total_roi" not in trainer.feature_cols
    assert "capital_flow_centrality" in trainer.feature_cols


def test_trainer_rejects_single_class_eval_split():
    import numpy as np
    from src.models.trainer import ModelTrainer

    trainer = ModelTrainer()
    trainer.y_train = np.array([0, 1])
    trainer.y_val = np.array([0, 1])
    trainer.y_test = np.array([0, 0])

    try:
        trainer._validate_label_splits()
    except ValueError as exc:
        assert "Single-class splits" in str(exc)
    else:
        raise AssertionError("Expected single-class split validation to fail")
