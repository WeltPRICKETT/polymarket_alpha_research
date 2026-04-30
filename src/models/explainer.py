"""
Author: AI Assistant
Date: 2026-03-18
Description: SHAP 可解释性分析模块。
使用 SHAP TreeExplainer 生成特征重要性排名和 Summary Plot，
验证研究假设 H2（哪些特征最能区分知情交易者）。
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

# Add project root to sys path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "features"
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Same feature columns as trainer.py
FEATURE_COLS = [
    "total_roi", "max_drawdown", "win_rate", "profit_loss_ratio",
    "early_entry_score", "contrarian_score", "information_ratio",
    "cross_market_diversification", "avg_holding_period", "trading_frequency",
    "capital_flow_centrality",
]


class ModelExplainer:
    """
    使用 SHAP 对最佳模型进行可解释性分析。
    """

    def __init__(self):
        self.model = None
        self.X_test = None
        self.shap_values = None

    def load(self):
        """加载最佳模型和测试数据。"""
        model_path = MODEL_DIR / "best_model.pkl"
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)

        df = pd.read_csv(DATA_DIR / "model_input.csv")
        test_df = df[df["is_train"] == False]
        self.X_test = test_df[FEATURE_COLS]
        logger.info(f"Loaded test data: {self.X_test.shape}")

    def compute_shap_values(self):
        """计算 SHAP 值。"""
        import shap
        logger.info("Computing SHAP values with TreeExplainer...")

        try:
            # TreeExplainer for tree-based models (XGBoost)
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(self.X_test)
        except Exception:
            # Fallback to KernelExplainer for non-tree models (e.g., LogReg Pipeline)
            logger.info("Falling back to KernelExplainer...")
            # Use a sample background
            bg = shap.sample(self.X_test, min(50, len(self.X_test)))
            if hasattr(self.model, "predict_proba"):
                explainer = shap.KernelExplainer(self.model.predict_proba, bg)
                self.shap_values = explainer.shap_values(self.X_test)[1]  # class 1
            else:
                explainer = shap.KernelExplainer(self.model.predict, bg)
                self.shap_values = explainer.shap_values(self.X_test)

        logger.info(f"SHAP values shape: {np.array(self.shap_values).shape}")

    def save_summary_plot(self):
        """生成并保存 SHAP Summary Plot。"""
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            self.shap_values,
            self.X_test,
            feature_names=FEATURE_COLS,
            show=False,
            plot_size=(10, 7),
        )
        plt.title("SHAP Feature Importance: Informed Trader Classification", fontsize=13)
        plt.tight_layout()

        plot_path = PLOTS_DIR / "shap_summary.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP summary plot to {plot_path}")

    def save_bar_plot(self):
        """生成并保存 SHAP Bar Plot (mean |SHAP|)。"""
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values,
            self.X_test,
            feature_names=FEATURE_COLS,
            plot_type="bar",
            show=False,
            plot_size=(10, 6),
        )
        plt.title("Mean |SHAP| Feature Importance", fontsize=13)
        plt.tight_layout()

        plot_path = PLOTS_DIR / "shap_bar.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP bar plot to {plot_path}")

    def export_feature_importance(self):
        """导出特征重要性排名至 CSV。"""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        shap_abs_mean = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "mean_abs_shap": shap_abs_mean,
        }).sort_values("mean_abs_shap", ascending=False)

        out_path = RESULTS_DIR / "feature_importance.csv"
        importance_df.to_csv(out_path, index=False)
        logger.info(f"Saved feature importance to {out_path}")
        logger.info(f"\nFeature Importance Ranking:\n{importance_df.to_string(index=False)}")

        return importance_df

    def run(self):
        """运行完整的可解释性分析。"""
        self.load()
        self.compute_shap_values()
        self.save_summary_plot()
        self.save_bar_plot()
        importance = self.export_feature_importance()

        logger.info("=" * 50)
        logger.info("SHAP Explainability Analysis Complete!")
        logger.info(f"Top 3 features: {importance['feature'].head(3).tolist()}")
        logger.info("=" * 50)


if __name__ == "__main__":
    explainer = ModelExplainer()
    explainer.run()
