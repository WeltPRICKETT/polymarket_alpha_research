"""
Author: AI Assistant
Date: 2026-03-26
Description: Academic-grade ML Research Engine for Polymarket Informed Trader Classification.

Models: Logistic Regression, Random Forest, XGBoost, LightGBM, Gaussian Naive Bayes
Evaluation: Confusion Matrix, ROC/PR/Calibration curves, Cross-Validation, McNemar's test
Outputs: All plots saved to results/plots/, tables to results/, model to models/artifacts/
"""

import sys
import json
import time
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from loguru import logger
from typing import Dict, List, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, 
    cross_validate, TimeSeriesSplit
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    log_loss, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from src.models.feature_sets import FEATURE_LEAKAGE_CLASS, get_feature_columns
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except (ImportError, Exception) as e:
    HAS_XGB = False
    logger.warning(f"XGBoost could not be loaded: {e}. Will skip XGBoost model.")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except (ImportError, Exception) as e:
    HAS_LGBM = False
    logger.warning(f"LightGBM could not be loaded: {e}. Will skip LightGBM model.")

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "features"
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

LABEL_COL = "Trader_Success_Rate"
ALLOW_SINGLE_CLASS_SPLITS = os.getenv("POLYMARKET_ALLOW_SINGLE_CLASS_SPLITS", "false").lower() == "true"
SKLEARN_N_JOBS = int(os.getenv("POLYMARKET_SKLEARN_N_JOBS", "1"))
SEARCH_N_ITER = int(os.getenv("POLYMARKET_SEARCH_N_ITER", "8"))

# ── Academic plot style ─────────────────────────────────────────────
def _set_plot_style():
    plt.rcParams.update({
        'axes.facecolor': '#ffffff',
        'figure.facecolor': '#ffffff',
        'axes.edgecolor': '#000000',
        'axes.linewidth': 1.0,
        'grid.color': '#dddddd',
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'text.color': '#000000',
        'axes.labelcolor': '#000000',
        'xtick.color': '#000000',
        'ytick.color': '#000000',
        'font.family': 'serif',
        'font.size': 11,
    })

ACADEMIC_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


class ValidationCalibratedModel:
    """Wrap a fitted model with a probability calibrator trained only on validation data."""

    def __init__(self, base_model, calibrator, method: str):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method
        self.classes_ = np.array([0, 1])

    def _raw_positive_proba(self, X):
        proba = self.base_model.predict_proba(X)
        classes = list(getattr(self.base_model, "classes_", [0, 1]))
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(X))

    def predict_proba(self, X):
        raw = self._raw_positive_proba(X)
        if self.method == "isotonic":
            calibrated = self.calibrator.predict(raw)
        else:
            calibrated = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ══════════════════════════════════════════════════════════════════
# ModelTrainer — Academic ML Research Engine
# ══════════════════════════════════════════════════════════════════

class ModelTrainer:
    """
    Trains 5 classification models, evaluates with academic-grade metrics,
    and exports all results as plots + CSV tables.
    """

    def __init__(self, data_path: str = None, feature_set: str = None):
        self.using_default_data_path = data_path is None
        self.data_path = data_path or str(DATA_DIR / "model_input.csv")
        self.feature_set = feature_set or os.getenv("POLYMARKET_FEATURE_SET", "live")
        self.feature_cols = get_feature_columns(self.feature_set)
        self.df = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.test_addresses = None
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.cv_results: List[Dict] = []
        self.split_counts: Dict[str, int] = {}
        self.label_counts_by_split: Dict[str, Dict[int, int]] = {}
        self.split_order_column = None
        self._status = {"state": "idle", "progress": 0, "message": ""}

    @property
    def status(self) -> dict:
        return self._status.copy()

    def _set_status(self, state: str, progress: int, message: str):
        self._status = {"state": state, "progress": progress, "message": message}

    # ── Data Loading ──────────────────────────────────────────────

    def load_data(self):
        """Load feature matrix and split by is_train column."""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)

        # Try different data files if primary is too small
        if self.using_default_data_path and len(self.df) < 20:
            alt_path = DATA_DIR / "model_input_top10.csv"
            if alt_path.exists():
                logger.info(f"Primary dataset too small. Loading {alt_path}")
                self.df = pd.read_csv(alt_path)

        if LABEL_COL not in self.df.columns:
            logger.warning(f"'{LABEL_COL}' not found. Computing from train-set Risk_Adjusted_Return...")
            train_rar = self.df.loc[self.df["is_train"] == True, "Risk_Adjusted_Return"]
            threshold = train_rar.quantile(0.8)
            self.df[LABEL_COL] = (self.df["Risk_Adjusted_Return"] >= threshold).astype(int)
            logger.info(f"Label threshold: {threshold:.4f}")

        if "split" in self.df.columns:
            train_df = self._sort_split_df(self.df[self.df["split"] == "train"].copy(), "train")
            val_df   = self._sort_split_df(self.df[self.df["split"] == "val"].copy(), "val")
            test_df  = self._sort_split_df(self.df[self.df["split"] == "test"].copy(), "test")
        else:
            # Backward compatibility
            logger.warning("'split' column missing. Using 'is_train' to create pseudo-val set.")
            train_full = self._sort_split_df(self.df[self.df["is_train"] == True].copy(), "train_full")
            test_df    = self._sort_split_df(self.df[self.df["is_train"] == False].copy(), "test")
            # Split train into train/val (temporal)
            split_idx = int(len(train_full) * 0.8)
            train_df = train_full.iloc[:split_idx]
            val_df   = train_full.iloc[split_idx:]

        missing_features = [col for col in self.feature_cols if col not in self.df.columns]
        if missing_features:
            raise ValueError(f"Feature set {self.feature_set!r} missing columns: {missing_features}")

        logger.info(f"Using feature_set={self.feature_set}: {self.feature_cols}")

        self.X_train = train_df[self.feature_cols].values
        self.y_train = train_df[LABEL_COL].values
        self.X_val   = val_df[self.feature_cols].values
        self.y_val   = val_df[LABEL_COL].values
        self.X_test  = test_df[self.feature_cols].values
        self.y_test  = test_df[LABEL_COL].values
        self.test_addresses = test_df["address"].values
        self.split_counts = {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        }
        self.label_counts_by_split = {
            "train": self._label_counts(self.y_train),
            "val": self._label_counts(self.y_val),
            "test": self._label_counts(self.y_test),
        }

        logger.info(f"Train: {len(self.X_train)} | Val: {len(self.X_val)} | Test: {len(self.X_test)}")

        logger.info(f"Label dist (train): {dict(zip(*np.unique(self.y_train, return_counts=True)))}")
        logger.info(f"Label dist (test):  {dict(zip(*np.unique(self.y_test, return_counts=True)))}")
        self._validate_label_splits()

    def _sort_split_df(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """Sort split rows chronologically before TimeSeriesSplit consumes arrays."""
        for col in ("first_trade_date", "timestamp"):
            if col in df.columns:
                sorted_df = df.assign(_sort_time=pd.to_datetime(df[col], errors="coerce"))
                sorted_df = sorted_df.sort_values(["_sort_time", "address"], kind="mergesort")
                self.split_order_column = col
                logger.info(f"Sorted {split_name} split by {col}.")
                return sorted_df.drop(columns=["_sort_time"])
        self.split_order_column = self.split_order_column or "address"
        logger.warning(f"No timestamp column available for {split_name}; sorting by address for deterministic order.")
        return df.sort_values("address", kind="mergesort") if "address" in df.columns else df

    @staticmethod
    def _label_counts(labels: np.ndarray) -> Dict[int, int]:
        values, counts = np.unique(labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def _validate_label_splits(self):
        """Prevent invalid binary evaluation when a split has only one class."""
        split_labels = {
            "train": self.y_train,
            "val": self.y_val,
            "test": self.y_test,
        }
        single_class = [
            split for split, labels in split_labels.items()
            if len(labels) == 0 or len(np.unique(labels)) < 2
        ]
        if single_class and not ALLOW_SINGLE_CLASS_SPLITS:
            raise ValueError(
                "Invalid label distribution for binary evaluation. "
                f"Single-class splits: {single_class}. "
                "Regenerate labels/splits or set POLYMARKET_ALLOW_SINGLE_CLASS_SPLITS=true for diagnostics only."
            )
        if single_class:
            logger.warning(f"Single-class splits allowed for diagnostics: {single_class}")

    # ── Model Definitions ─────────────────────────────────────────

    def _build_models(self) -> Dict[str, Tuple[Any, Dict]]:
        """Returns dict of model_name -> (estimator, param_grid)."""
        n_neg = (self.y_train == 0).sum()
        n_pos = (self.y_train == 1).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        models = {
            "Logistic Regression": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")),
                ]),
                {
                    "lr__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    "lr__penalty": ["l2"],
                    "lr__solver": ["lbfgs"]
                },
            ),
            "Random Forest": (
                RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=SKLEARN_N_JOBS),
                {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [3, 5, 7, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None]
                },
            ),
        }

        if HAS_XGB:
            models["XGBoost"] = (
                XGBClassifier(
                    eval_metric="logloss",
                    random_state=42, scale_pos_weight=spw,
                    n_jobs=SKLEARN_N_JOBS,
                ),
                {
                    "n_estimators": [100, 200, 500, 1000],
                    "max_depth": [3, 5, 7, 9],
                    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                    "min_child_weight": [1, 3, 5, 7],
                    "gamma": [0, 0.1, 0.2, 0.4],
                    "reg_alpha": [0, 0.01, 0.1, 1.0],
                    "reg_lambda": [0.1, 1.0, 5.0, 10.0]
                },
            )

        models["Gaussian NB"] = (
            Pipeline([
                ("scaler", StandardScaler()),
                ("nb", GaussianNB()),
            ]),
            {},
        )

        if HAS_LGBM:
            models["LightGBM"] = (
                LGBMClassifier(
                    random_state=42, scale_pos_weight=spw,
                    verbose=-1, n_jobs=SKLEARN_N_JOBS,
                ),
                {
                    "n_estimators": [100, 200, 500, 1000],
                    "max_depth": [-1, 3, 5, 7, 10],
                    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
                    "num_leaves": [15, 31, 63, 127],
                    "feature_fraction": [0.7, 0.8, 0.9, 1.0],
                    "bagging_fraction": [0.7, 0.8, 0.9, 1.0],
                    "min_child_samples": [10, 20, 30, 50],
                    "reg_alpha": [0, 0.01, 0.1, 1.0],
                    "reg_lambda": [0, 1.0, 5.0, 10.0]
                },
            )

        return models

    def _build_baseline_models(self) -> Dict[str, Any]:
        """Simple baselines that real models must beat."""
        return {
            "Baseline: Majority Class": DummyClassifier(strategy="most_frequent"),
            "Baseline: Stratified Random": DummyClassifier(strategy="stratified", random_state=42),
        }

    # ── Training ──────────────────────────────────────────────────

    def train_all(self):
        """Train all models with GridSearchCV and evaluate."""
        self._train_baselines()

        model_defs = self._build_models()
        total = len(model_defs)

        for i, (name, (estimator, param_grid)) in enumerate(model_defs.items()):
            progress = int((i / total) * 60) + 10
            self._set_status("training", progress, f"Training {name} ({i+1}/{total})")
            logger.info("=" * 60)
            logger.info(f"Training [{i+1}/{total}]: {name}")
            logger.info("=" * 60)

            start = time.time()

            if param_grid:
                # Decide search method: use RandomizedSearch for large grids
                is_complex = name in ["Random Forest", "XGBoost", "LightGBM"]
                cv = TimeSeriesSplit(n_splits=5)
                
                # Keep search fit parameters version-neutral. Recent XGBoost and
                # LightGBM sklearn wrappers reject legacy early_stopping_rounds
                # in fit(); production runs can add version-pinned callbacks.
                fit_params = {}

                if is_complex:
                    # Use RandomizedSearchCV for large grids
                    search = RandomizedSearchCV(
                        estimator=estimator,
                        param_distributions=param_grid,
                        n_iter=SEARCH_N_ITER,
                        cv=cv,
                        scoring="roc_auc",
                        n_jobs=SKLEARN_N_JOBS,
                        random_state=42,
                        return_train_score=True
                    )
                else:
                    search = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                        cv=cv,
                        scoring="roc_auc",
                        n_jobs=SKLEARN_N_JOBS,
                        return_train_score=True
                    )
                try:
                    logger.info(f"Running {'Randomized' if is_complex else 'Grid'} search for {name}...")
                    search.fit(self.X_train, self.y_train, **fit_params)
                    
                    best_model = search.best_estimator_
                    best_cv_score = search.best_score_
                    best_params = search.best_params_
                    logger.info(f"Best params: {best_params}")
                    logger.info(f"Best CV AUC-ROC: {best_cv_score:.4f}")
                except Exception as e:
                    logger.error(f"❌ Failed to train {name}: {e}")
                    if "libomp" in str(e) or "Library not loaded" in str(e):
                        logger.warning(f"⚠️ {name} requires libomp which is missing. Skipping.")
                    continue
            else:
                best_model = estimator
                best_model.fit(self.X_train, self.y_train)
                best_cv_score = 0.0
                best_params = {}

            # Cross-validation detailed scores on the uncalibrated fitted estimator.
            cv_detail = self._cross_validate(name, best_model)
            best_model, calibration_info = self._calibrate_on_validation(name, best_model)

            elapsed = time.time() - start
            self.models[name] = best_model

            # Evaluate
            metrics = self._evaluate(name, best_model)
            metrics["cv_auc"] = best_cv_score
            metrics["best_params"] = best_params
            metrics["train_time_sec"] = round(elapsed, 2)
            metrics["calibration"] = calibration_info

            metrics["cv_detail"] = cv_detail

            # P2.2: Overfitting Detection (Train vs Val)
            train_auc = self._safe_auc(self.y_train, self._positive_proba(best_model, self.X_train))
            val_auc = self._safe_auc(self.y_val, self._positive_proba(best_model, self.X_val))
            test_auc = metrics["auc_roc"]
            
            metrics["train_auc"] = train_auc
            metrics["val_auc"] = val_auc

            if train_auc - val_auc > 0.05:
                logger.warning(f"⚠️ OVERFITTING DETECTED (Val) for {name}: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}")
                metrics["overfit_warning_val"] = True
            else:
                metrics["overfit_warning_val"] = False

            if train_auc - test_auc > 0.05:
                logger.warning(f"⚠️ OVERFITTING DETECTED (Test) for {name}: Train AUC={train_auc:.4f}, Test AUC={test_auc:.4f}")
                metrics["overfit_warning_test"] = True
            else:
                metrics["overfit_warning_test"] = False

            self.results[name] = metrics
            logger.info(f"  Training time: {elapsed:.1f}s")

    def _train_baselines(self):
        """Train and evaluate non-ML baselines before tuned models."""
        for name, model in self._build_baseline_models().items():
            logger.info("=" * 60)
            logger.info(f"Training baseline: {name}")
            logger.info("=" * 60)
            start = time.time()
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            metrics = self._evaluate(name, model)
            metrics["cv_auc"] = 0.5
            metrics["best_params"] = {}
            metrics["train_time_sec"] = round(time.time() - start, 2)
            metrics["calibration"] = {"method": "none", "data": "not_applicable"}
            metrics["cv_detail"] = {}
            metrics["train_auc"] = self._safe_auc(self.y_train, self._positive_proba(model, self.X_train))
            metrics["val_auc"] = self._safe_auc(self.y_val, self._positive_proba(model, self.X_val))
            metrics["overfit_warning_val"] = False
            metrics["overfit_warning_test"] = False
            self.results[name] = metrics

    def _calibrate_on_validation(self, name: str, model):
        """Calibrate a fitted estimator on the independent validation split."""
        if len(np.unique(self.y_val)) < 2:
            logger.warning(f"Skipping calibration for {name}: validation split has one class.")
            return model, {"method": "none", "data": "skipped_single_class_validation"}

        method = "isotonic" if min(np.bincount(self.y_val.astype(int), minlength=2)) >= 30 else "sigmoid"
        logger.info(f"Calibrating probabilities on validation split (method={method})...")
        raw_val = self._positive_proba(model, self.X_val)
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_val, self.y_val)
        else:
            calibrator = LogisticRegression(max_iter=1000, random_state=42)
            calibrator.fit(raw_val.reshape(-1, 1), self.y_val)
        calibrated_model = ValidationCalibratedModel(model, calibrator, method)
        return calibrated_model, {
            "method": method,
            "data": "validation",
            "validation_samples": int(len(self.y_val)),
        }

    def _positive_proba(self, model, X: np.ndarray) -> np.ndarray:
        proba = model.predict_proba(X)
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(X))

    @staticmethod
    def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_prob))

    def _evaluate(self, name: str, model) -> Dict:
        """Evaluate model on test set, return metrics dict."""
        y_pred = model.predict(self.X_test)
        y_prob = self._positive_proba(model, self.X_test)

        n_classes = len(np.unique(self.y_test))
        
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        auc_roc = 0.5
        avg_prec = 0.0
        logloss = 1.0
        
        if n_classes > 1:
            try:
                auc_roc = roc_auc_score(self.y_test, y_prob)
                avg_prec = average_precision_score(self.y_test, y_prob)
                logloss = log_loss(self.y_test, y_prob, labels=[0, 1])
            except:
                pass
        else:
            logger.warning(f"  Note: Only one class present in test set for {name}. Some metrics (AUC, LogLoss) may be placeholder.")
            try:
                # Still try log_loss with explicit labels
                logloss = log_loss(self.y_test, y_prob, labels=[0, 1])
            except:
                pass

        brier = brier_score_loss(self.y_test, y_prob)

        logger.info(f"\n--- {name} Test Results ---")
        logger.info(f"  Accuracy:  {acc:.4f}  |  AUC-ROC: {auc_roc:.4f}")
        logger.info(f"  Precision: {prec:.4f}  |  Recall:  {rec:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}  |  Avg Prec: {avg_prec:.4f}")
        logger.info(f"  Log Loss:  {logloss:.4f}  |  Brier:   {brier:.4f}")
        
        # Handle classification report for single class
        target_names = ['Noise', 'Informed']
        if n_classes == 1:
            present_label = int(self.y_test[0])
            target_names = [target_names[present_label]]
            
        try:
            report = classification_report(self.y_test, y_pred, labels=np.unique(self.y_test), target_names=target_names)
            logger.info(f"\n{report}")
        except:
            logger.info("\n(Classification report unavailable due to single-class test set)")

        return {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "auc_roc": auc_roc, "avg_precision": avg_prec,
            "log_loss": logloss, "brier_score": brier,
            "y_pred": y_pred, "y_prob": y_prob,
        }

    def _cross_validate(self, name: str, model) -> Dict:
        """Run 5-fold TimeSeries split cross-validation."""
        cv = TimeSeriesSplit(n_splits=5)
        try:
            scores = cross_validate(
                model, self.X_train, self.y_train, cv=cv,
                scoring=["accuracy", "roc_auc", "f1", "precision", "recall"],
                n_jobs=SKLEARN_N_JOBS, return_train_score=False,
            )
            detail = {
                "accuracy_mean": scores["test_accuracy"].mean(),
                "accuracy_std": scores["test_accuracy"].std(),
                "auc_mean": scores["test_roc_auc"].mean(),
                "auc_std": scores["test_roc_auc"].std(),
                "f1_mean": scores["test_f1"].mean(),
                "f1_std": scores["test_f1"].std(),
                "precision_mean": scores["test_precision"].mean(),
                "recall_mean": scores["test_recall"].mean(),
            }
            self.cv_results.append({"model": name, **detail})
            return detail
        except Exception as e:
            logger.warning(f"CV failed for {name}: {e}")
            return {}

    # ── Plotting ──────────────────────────────────────────────────

    def save_all_plots(self):
        """Generate and save all academic evaluation plots."""
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        _set_plot_style()

        n_classes = len(np.unique(self.y_test))
        if n_classes < 2:
            logger.warning("⚠️ Test set contains only one class. Evaluation plots (ROC, PR, CM) will be skipped as they require both classes.")
            self._set_status("plotting", 95, "Generating model comparison bar")
            self._plot_model_comparison_bar()
            return

        self._set_status("plotting", 70, "Generating confusion matrices")
        self._plot_confusion_matrices()

        self._set_status("plotting", 75, "Generating ROC curves")
        self._plot_roc_curves()

        self._set_status("plotting", 80, "Generating PR curves")
        self._plot_pr_curves()

        self._set_status("plotting", 85, "Generating calibration plot")
        self._plot_calibration()

        self._set_status("plotting", 95, "Generating model comparison bar")
        self._plot_model_comparison_bar()

        self._set_status("plotting", 88, "Generating model comparison chart")
        self._plot_model_comparison_bar()

    def _plot_confusion_matrices(self):
        """Save confusion matrix heatmap for each model."""
        n_models = len(self.results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (name, model) in enumerate(self.models.items()):
            ax = axes[i]
            y_pred = model.predict(self.X_test)
            # Robust CM even if labels are missing
            cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.set_title(name, color='#000000', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted', color='#000000')
            ax.set_ylabel('Actual', color='#000000')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Noise', 'Informed'])
            ax.set_yticklabels(['Noise', 'Informed'])

            # Annotate cells
            for r in range(2):
                for c in range(2):
                    color = 'white' if cm[r, c] > cm.max() / 2 else 'black'
                    ax.text(c, r, str(cm[r, c]), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=color)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Confusion Matrices", color='#000000',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = PLOTS_DIR / "confusion_matrices.png"
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
        plt.close()
        logger.info(f"Saved confusion matrices to {path}")

    def _plot_roc_curves(self):
        """ROC curves for all models on one plot."""
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, (name, res) in enumerate(self.results.items()):
            fpr, tpr, _ = roc_curve(self.y_test, res["y_prob"])
            roc_auc = res["auc_roc"]
            color = ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{name} (AUC={roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], '--', color='grey', alpha=0.8, linewidth=1.5,
                label='Random (AUC=0.500)')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=16,
                     color='#000000', fontweight='bold')
        leg = ax.legend(loc='lower right', frameon=True, edgecolor='#000000',
                       facecolor='#ffffff', fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        path = PLOTS_DIR / "roc_comparison.png"
        plt.savefig(path, dpi=300, facecolor='#ffffff', bbox_inches='tight')
        plt.close()
        logger.info(f"Saved ROC comparison to {path}")

    def _plot_pr_curves(self):
        """Precision-Recall curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, (name, res) in enumerate(self.results.items()):
            precision, recall, _ = precision_recall_curve(self.y_test, res["y_prob"])
            ap = res["avg_precision"]
            color = ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]
            ax.plot(recall, precision, color=color, linewidth=2,
                    label=f'{name} (AP={ap:.3f})')

        # Baseline
        baseline = self.y_test.sum() / len(self.y_test)
        ax.axhline(y=baseline, color='grey', linestyle='--', alpha=0.8, linewidth=1.5,
                   label=f'Baseline ({baseline:.3f})')

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=16,
                     color='#000000', fontweight='bold')
        leg = ax.legend(loc='upper right', frameon=True, edgecolor='#000000',
                       facecolor='#ffffff', fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        path = PLOTS_DIR / "pr_comparison.png"
        plt.savefig(path, dpi=300, facecolor='#ffffff', bbox_inches='tight')
        plt.close()
        logger.info(f"Saved PR comparison to {path}")

    def _plot_calibration(self):
        """Calibration curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot([0, 1], [0, 1], '--', color='grey', alpha=0.8, linewidth=1.5, label='Perfectly Calibrated')

        for i, (name, res) in enumerate(self.results.items()):
            try:
                prob_true, prob_pred = calibration_curve(
                    self.y_test, res["y_prob"], n_bins=8, strategy='uniform'
                )
                color = ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]
                brier = res["brier_score"]
                ax.plot(prob_pred, prob_true, 's-', color=color, linewidth=2,
                        markersize=6, label=f'{name} (Brier={brier:.3f})')
            except Exception:
                continue

        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Calibration Curves', fontsize=16,
                     color='#000000', fontweight='bold')
        leg = ax.legend(loc='upper left', frameon=True, edgecolor='#000000',
                       facecolor='#ffffff', fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        path = PLOTS_DIR / "calibration.png"
        plt.savefig(path, dpi=300, facecolor='#ffffff', bbox_inches='tight')
        plt.close()
        logger.info(f"Saved calibration plot to {path}")

    def _plot_model_comparison_bar(self):
        """Bar chart comparing key metrics across all models."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        model_names = list(self.results.keys())
        metrics_list = [
            ("AUC-ROC", [self.results[n]["auc_roc"] for n in model_names]),
            ("F1-Score", [self.results[n]["f1"] for n in model_names]),
            ("Avg Precision", [self.results[n]["avg_precision"] for n in model_names]),
        ]

        for ax, (metric_name, values) in zip(axes, metrics_list):
            colors = [ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)] for i in range(len(model_names))]
            bars = ax.bar(range(len(model_names)), values, color=colors, alpha=0.8,
                         edgecolor='#000000', linewidth=1)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=10)
            ax.set_title(metric_name, color='#000000', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1.05)

            # Value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', color='#000000',
                       fontsize=10)

        fig.suptitle('Model Performance Summary', color='#000000',
                    fontsize=18, fontweight='bold')
        plt.tight_layout()

        path = PLOTS_DIR / "model_comparison_bar.png"
        plt.savefig(path, dpi=300, facecolor='#ffffff', bbox_inches='tight')
        plt.close()
        logger.info(f"Saved model comparison bar chart to {path}")

    # ── Tables Export ─────────────────────────────────────────────

    def save_all_tables(self):
        """Export CSV tables and JSON report."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        self._set_status("exporting", 90, "Exporting results tables")

        # 1. Model comparison table
        rows = []
        for name, res in self.results.items():
            rows.append({
                "Model": name,
                "Model Type": "baseline" if name.startswith("Baseline:") else "model",
                "Accuracy": round(res["accuracy"], 4),
                "Precision": round(res["precision"], 4),
                "Recall": round(res["recall"], 4),
                "F1-Score": round(res["f1"], 4),
                "AUC-ROC": round(res["auc_roc"], 4),
                "Avg Precision": round(res["avg_precision"], 4),
                "Log Loss": round(res["log_loss"], 4),
                "Brier Score": round(res["brier_score"], 4),
                "CV AUC (5-fold)": round(res.get("cv_auc", 0), 4),
                "Train Time (s)": res.get("train_time_sec", 0),
            })
        comp_df = pd.DataFrame(rows)
        comp_path = RESULTS_DIR / "model_comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        logger.info(f"Saved model comparison to {comp_path}")
        logger.info(f"\n{comp_df.to_string(index=False)}")

        # 2. Cross-validation results
        if self.cv_results:
            cv_df = pd.DataFrame(self.cv_results)
            cv_path = RESULTS_DIR / "cv_results.csv"
            cv_df.to_csv(cv_path, index=False)
            logger.info(f"Saved CV results to {cv_path}")

        # 3. Statistical significance: McNemar's test (pairwise)
        self._mcnemar_tests()

        # 4. Full training report JSON
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_path": self.data_path,
            "split_order_column": self.split_order_column,
            "split_counts": self.split_counts,
            "label_counts_by_split": self.label_counts_by_split,
            "train_samples": len(self.X_train),
            "val_samples": len(self.X_val),
            "test_samples": len(self.X_test),
            "feature_set": self.feature_set,
            "features": self.feature_cols,
            "feature_leakage_class": {
                col: FEATURE_LEAKAGE_CLASS.get(col, "unknown")
                for col in self.feature_cols
            },
            "label_audit": self._load_label_audit_summary(),
            "baselines": list(self._build_baseline_models().keys()),
            "models": {},
        }
        for name, res in self.results.items():
            report["models"][name] = {
                k: (v if not isinstance(v, np.ndarray) else v.tolist())
                for k, v in res.items()
                if k not in ("y_pred", "y_prob", "cv_detail")
            }
        report_path = RESULTS_DIR / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved training report to {report_path}")

    def _load_label_audit_summary(self) -> Dict:
        summary_path = RESULTS_DIR / "label_summary.json"
        if not summary_path.exists():
            return {}
        try:
            with open(summary_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read label_summary.json: {e}")
            return {}

    def _mcnemar_tests(self):
        """Pairwise McNemar's test for statistical significance."""
        model_names = list(self.results.keys())
        if len(model_names) < 2:
            return

        rows = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name_a, name_b = model_names[i], model_names[j]
                pred_a = self.results[name_a]["y_pred"]
                pred_b = self.results[name_b]["y_pred"]

                # Contingency: both correct, a correct b wrong, a wrong b correct, both wrong
                both_correct = ((pred_a == self.y_test) & (pred_b == self.y_test)).sum()
                a_only = ((pred_a == self.y_test) & (pred_b != self.y_test)).sum()
                b_only = ((pred_a != self.y_test) & (pred_b == self.y_test)).sum()
                both_wrong = ((pred_a != self.y_test) & (pred_b != self.y_test)).sum()

                # McNemar's chi-squared (with continuity correction)
                n = a_only + b_only
                if n > 0:
                    chi2 = ((abs(a_only - b_only) - 1) ** 2) / n
                    # p-value from chi2 distribution with 1 df
                    from scipy import stats
                    p_value = 1 - stats.chi2.cdf(chi2, df=1)
                else:
                    chi2 = 0.0
                    p_value = 1.0

                significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))

                rows.append({
                    "Model A": name_a,
                    "Model B": name_b,
                    "A_correct_B_wrong": a_only,
                    "A_wrong_B_correct": b_only,
                    "Chi2": round(chi2, 4),
                    "p_value": round(p_value, 6),
                    "Significance": significance,
                })

        mcnemar_df = pd.DataFrame(rows)
        mcnemar_path = RESULTS_DIR / "mcnemar_tests.csv"
        mcnemar_df.to_csv(mcnemar_path, index=False)
        logger.info(f"Saved McNemar's tests to {mcnemar_path}")
        logger.info(f"\n{mcnemar_df.to_string(index=False)}")

    # ── Model Save ────────────────────────────────────────────────

    def save_best_model(self) -> str:
        """Save the best model (highest AUC-ROC) to disk."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        best_name = max(self.results, key=lambda k: self.results[k]["auc_roc"])
        best_model = self.models[best_name]
        best_auc = self.results[best_name]["auc_roc"]

        model_path = MODEL_DIR / "best_model.pkl"
        joblib.dump(best_model, model_path)
        logger.info(f"Saved best model ({best_name}, AUC={best_auc:.4f}) to {model_path}")

        # Data snapshot hash for traceability
        import hashlib
        data_file = Path(self.data_path)
        data_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()[:16] if data_file.exists() else "unknown"
        label_summary = self._load_label_audit_summary()

        meta = {
            "model_name": best_name,
            "auc_roc": best_auc,
            "accuracy": self.results[best_name]["accuracy"],
            "precision": self.results[best_name]["precision"],
            "recall": self.results[best_name]["recall"],
            "f1": self.results[best_name]["f1"],
            "avg_precision": self.results[best_name]["avg_precision"],
            "brier_score": self.results[best_name]["brier_score"],
            "feature_set": self.feature_set,
            "feature_columns": self.feature_cols,
            "data_snapshot_hash": data_hash,
            "label_strategy": label_summary.get("label_strategy", "unknown"),
            "training_date": pd.Timestamp.now().isoformat(),
            "train_samples": len(self.X_train),
            "val_samples": len(self.X_val),
            "test_samples": len(self.X_test),
        }
        meta_path = MODEL_DIR / "model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return best_name

    def export_informed_traders(self):
        """Export informed trader predictions from best model."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        best_name = max(self.results, key=lambda k: self.results[k]["auc_roc"])
        result = self.results[best_name]

        export_df = pd.DataFrame({
            "address": self.test_addresses,
            "predicted_label": result["y_pred"],
            "predicted_probability": result["y_prob"],
            "actual_label": self.y_test,
        }).sort_values("predicted_probability", ascending=False)

        out_path = RESULTS_DIR / "informed_traders.csv"
        export_df.to_csv(out_path, index=False)

        n_informed = (export_df["predicted_label"] == 1).sum()
        logger.info(f"Exported {len(export_df)} predictions ({n_informed} informed) to {out_path}")
        return export_df

    # ── Full Pipeline ─────────────────────────────────────────────

    def run(self) -> Dict:
        """Execute the complete ML research pipeline."""
        self._set_status("running", 5, "Loading data")
        self.load_data()

        self._set_status("training", 10, "Training models")
        self.train_all()

        self.save_all_plots()
        self.save_all_tables()

        self._set_status("saving", 95, "Saving best model")
        best_name = self.save_best_model()
        self.export_informed_traders()

        logger.info("=" * 60)
        logger.info("ML RESEARCH PIPELINE COMPLETE")
        logger.info(f"Best model: {best_name} (AUC-ROC: {self.results[best_name]['auc_roc']:.4f})")
        logger.info(f"Results saved to: {RESULTS_DIR}")
        logger.info(f"Plots saved to:   {PLOTS_DIR}")
        logger.info("=" * 60)

        self._set_status("done", 100, f"Complete — Best: {best_name}")

        # Return serializable summary
        summary = {}
        for name, res in self.results.items():
            summary[name] = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in res.items()
                if k not in ("y_pred", "y_prob", "cv_detail", "best_params")
            }
        return summary


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
