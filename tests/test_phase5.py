import pandas as pd


LIVE_FEATURES = [
    "early_entry_score",
    "contrarian_score",
    "information_ratio",
    "cross_market_diversification",
    "avg_holding_period",
    "trading_frequency",
    "capital_flow_centrality",
]


def _row(address, split, label, first_trade_date, marker):
    row = {
        "address": address,
        "split": split,
        "is_train": split == "train",
        "first_trade_date": first_trade_date,
        "Trader_Success_Rate": label,
    }
    for i, col in enumerate(LIVE_FEATURES):
        row[col] = marker if i == 0 else marker + i / 100
    return row


def test_trainer_sorts_splits_chronologically(tmp_path):
    from src.models.trainer import ModelTrainer

    df = pd.DataFrame([
        _row("late_train", "train", 0, "2026-01-03", 30),
        _row("early_train", "train", 1, "2026-01-01", 10),
        _row("mid_train", "train", 0, "2026-01-02", 20),
        _row("mid_train_pos", "train", 1, "2026-01-02", 21),
        _row("val_a", "val", 0, "2026-01-04", 40),
        _row("val_b", "val", 1, "2026-01-05", 50),
        _row("test_a", "test", 0, "2026-01-06", 60),
        _row("test_b", "test", 1, "2026-01-07", 70),
    ])
    path = tmp_path / "model_input.csv"
    df.to_csv(path, index=False)

    trainer = ModelTrainer(data_path=str(path))
    trainer.load_data()

    assert trainer.split_order_column == "first_trade_date"
    assert trainer.X_train[:, 0].tolist() == [10, 20, 21, 30]
    assert trainer.split_counts == {"train": 4, "val": 2, "test": 2}


def test_baseline_models_are_evaluated(tmp_path):
    from src.models.trainer import ModelTrainer

    rows = []
    for i in range(8):
        split = "train" if i < 4 else ("val" if i < 6 else "test")
        rows.append(_row(f"a{i}", split, i % 2, f"2026-01-{i + 1:02d}", i + 1))
    path = tmp_path / "model_input.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    trainer = ModelTrainer(data_path=str(path))
    trainer.load_data()
    trainer._train_baselines()

    assert "Baseline: Majority Class" in trainer.results
    assert "Baseline: Stratified Random" in trainer.results
    assert trainer.results["Baseline: Majority Class"]["calibration"]["method"] == "none"


def test_validation_calibration_uses_validation_split(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from src.models.trainer import ModelTrainer

    rows = []
    for i in range(12):
        split = "train" if i < 6 else ("val" if i < 9 else "test")
        rows.append(_row(f"a{i}", split, i % 2, f"2026-01-{i + 1:02d}", i + 1))
    path = tmp_path / "model_input.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    trainer = ModelTrainer(data_path=str(path))
    trainer.load_data()
    model = LogisticRegression().fit(trainer.X_train, trainer.y_train)
    calibrated, info = trainer._calibrate_on_validation("Logistic Regression", model)

    assert info["data"] == "validation"
    assert info["validation_samples"] == 3
    assert hasattr(calibrated, "predict_proba")
