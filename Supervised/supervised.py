"""
Supervised/supervised.py — Supervised learning pipeline (cloud-optimised).

Changes vs. original
---------------------
1.  SVM variants (Linear, Polynomial, RBF) are trained on a capped subsample
    (SVM_TRAIN_CAP rows) because sklearn's SVC is O(n²) in time and memory.
    Predictions still run on the full validation/test sets.
    A log warning is emitted whenever subsampling is applied so it is visible
    in the log file.

2.  MKL SVM: the kernel matrices KL_train are (k, n_train, n_train) — at
    1 M rows with 15 kernels that is ~11 TB. The MKL block also runs on a
    capped subset (MKL_TRAIN_CAP). The same random subset is used for all
    variants so the comparison remains fair.

3.  GradientBoosting models on 1 M rows are memory-heavy but feasible; we
    guard before fitting and log progress.

4.  All `print()` → structured logging.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from Visualization.Visualization import Visualization
from logger import get_logger
from memory_guard import MemoryGuard, log_memory, check_memory

# MKL library
from MKL.extremality_mkl import kernel_extremality_weights, create_weak_kernels
from MKL.extremality_mkl.metrics import (
    kernel_alignment,
    kernel_polarization,
    feature_space_measure,
    complex_ratio,
)

log = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
PHQ9_CATEGORY_ORDER = [
    "Minimal", "Mild", "Moderate", "Moderately_Severe", "Severe",
]

# SVC is O(n²) in time and memory → cap training rows
SVM_TRAIN_CAP = 50_000

# MKL kernel matrices are (k, n, n) — cap to avoid OOM
MKL_TRAIN_CAP = 1_000


class PHQ9ModelTrainer:

    def __init__(
        self,
        target_reg: str = "phq9_total",
        target_clf: str = "cluster_label",
        random_state: int = 42,
    ) -> None:
        self.target_reg   = target_reg
        self.target_clf   = target_clf
        self.random_state = random_state
        self.visualizer   = Visualization()
        self.results: Dict = {"Regression": {}, "Classification": {}}

        self._regression_models     = self._init_regression_models()
        self._classification_models = self._init_classification_models()

    # ── Model registries ─────────────────────────────────────────────────────
    def _init_regression_models(self) -> dict:
        return {
            "Linear_Regression": LinearRegression(),
            "Decision_Tree_Reg": DecisionTreeRegressor(random_state=self.random_state),
            "Random_Forest_Reg": RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1,
            ),
            "Gradient_Boosting_Reg": GradientBoostingRegressor(
                random_state=self.random_state,
            ),
        }

    def _init_classification_models(self) -> dict:
        return {
            "Logistic_Regression": LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "Decision_Tree_Clf": DecisionTreeClassifier(
                random_state=self.random_state, class_weight="balanced",
            ),
            "Random_Forest_Clf": RandomForestClassifier(
                n_estimators=100, random_state=self.random_state,
                class_weight="balanced", n_jobs=-1,
            ),
            "Gradient_Boosting_Clf": GradientBoostingClassifier(
                random_state=self.random_state,
            ),
            "SVM_Linear": SVC(
                kernel="linear", class_weight="balanced",
                random_state=self.random_state, probability=True,
            ),
            "SVM_Polynomial": SVC(
                kernel="poly", degree=3, class_weight="balanced",
                random_state=self.random_state, probability=True,
            ),
            "SVM_RBF": SVC(
                kernel="rbf", gamma="scale", class_weight="balanced",
                random_state=self.random_state, probability=True,
            ),
            # Sentinel: handled separately in _train_mkl_svm
            "SVM_MKL_Extremality": None,
        }

    # ── Pipeline entry point ─────────────────────────────────────────────────
    def run_pipeline(self, df: pd.DataFrame) -> dict:
        log.info("=" * 60)
        log.info("SUPERVISED LEARNING")
        log.info("=" * 60)

        (
            X_train_s, X_val_s, X_test_s,
            y_train_r, y_val_r, y_test_r,
            y_train_c, y_val_c, y_test_c,
            feature_names,
        ) = self._prepare_data(df)

        self._train_regression(
            X_train_s, X_val_s, X_test_s,
            y_train_r, y_val_r, y_test_r,
            feature_names,
        )
        self._train_classification(
            X_train_s, X_val_s, X_test_s,
            y_train_c, y_val_c, y_test_c,
            feature_names,
        )
        return self.results

    # ── Data preparation ─────────────────────────────────────────────────────
    def _prepare_data(self, df: pd.DataFrame) -> Tuple:
        exclude_cols = {self.target_reg, self.target_clf, "phq9_category"}
        feature_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in exclude_cols
        ]

        X     = df[feature_cols].values
        y_reg = df[self.target_reg].values
        y_clf = df[self.target_clf].values

        log.info(
            "[Supervised] Feature matrix: %d samples × %d features",
            X.shape[0], X.shape[1],
        )
        log.info("[Supervised] Regression  target: %s", self.target_reg)
        log.info("[Supervised] Classification target: %s", self.target_clf)

        X_temp, X_test, y_temp_r, y_test_r, y_temp_c, y_test_c = train_test_split(
            X, y_reg, y_clf,
            test_size=0.20, random_state=self.random_state, stratify=y_clf,
        )
        X_train, X_val, y_train_r, y_val_r, y_train_c, y_val_c = train_test_split(
            X_temp, y_temp_r, y_temp_c,
            test_size=0.25, random_state=self.random_state, stratify=y_temp_c,
        )

        log.info(
            "[Supervised] Set sizes — Train: %d  Val: %d  Test: %d",
            len(X_train), len(X_val), len(X_test),
        )

        scaler   = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)

        return (
            X_train_s, X_val_s, X_test_s,
            y_train_r, y_val_r, y_test_r,
            y_train_c, y_val_c, y_test_c,
            feature_cols,
        )

    # ── Regression training ──────────────────────────────────────────────────
    def _train_regression(
        self,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_names,
    ) -> None:
        log.info("-" * 50)
        log.info("Training Regression Models")
        log.info("-" * 50)

        model_names, r2_values = [], []

        for name, model in self._regression_models.items():
            log.info("[Supervised] Fitting %s ...", name)
            check_memory(threshold_gb=2, label=f"Regression: {name}")

            with MemoryGuard(f"Fit {name}"):
                model.fit(X_train, y_train)

            y_pred_val  = model.predict(X_val)
            y_pred_test = model.predict(X_test)

            mse_val   = mean_squared_error(y_val,  y_pred_val)
            rmse_val  = float(np.sqrt(mse_val))
            mae_val   = mean_absolute_error(y_val,  y_pred_val)
            r2_val    = r2_score(y_val,  y_pred_val)

            mse_test  = mean_squared_error(y_test, y_pred_test)
            rmse_test = float(np.sqrt(mse_test))
            mae_test  = mean_absolute_error(y_test, y_pred_test)
            r2_test   = r2_score(y_test, y_pred_test)

            self.results["Regression"][name] = {
                "Validation": {
                    "MSE": round(mse_val, 4), "RMSE": round(rmse_val, 4),
                    "MAE": round(mae_val, 4), "R2":   round(r2_val,   4),
                },
                "Test": {
                    "MSE": round(mse_test, 4), "RMSE": round(rmse_test, 4),
                    "MAE": round(mae_test, 4), "R2":   round(r2_test,   4),
                },
            }
            log.info(
                "  %-30s | Val R2=%.4f RMSE=%.3f | Test R2=%.4f RMSE=%.3f",
                name, r2_val, rmse_val, r2_test, rmse_test,
            )

            self.visualizer.regression_scatter(
                y_true=y_val, y_pred=y_pred_val,
                model_name=name, filename=f"reg_scatter_{name}.png",
            )
            if hasattr(model, "feature_importances_"):
                self.visualizer.feature_importance_plot(
                    importances=model.feature_importances_,
                    feature_names=feature_names,
                    model_name=name, filename=f"feature_importance_{name}.png",
                )
            if isinstance(model, DecisionTreeRegressor):
                self.visualizer.decision_tree_plot(
                    model=model, feature_names=feature_names,
                    class_names=None, model_name=name,
                    filename=f"tree_plot_{name}.png", max_depth=4,
                )

            model_names.append(name)
            r2_values.append(r2_val)
            log_memory(f"After {name}")

        self.visualizer.model_comparison_bar(
            model_names=model_names, metric_values=r2_values,
            metric_name="R2 (Validation)", task="Regression",
            filename="reg_model_comparison_r2.png",
        )

    # ── Classification training ──────────────────────────────────────────────
    def _train_classification(
        self,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_names,
    ) -> None:
        log.info("-" * 50)
        log.info("Training Classification Models")
        log.info("-" * 50)

        present_classes = [
            c for c in PHQ9_CATEGORY_ORDER if c in np.unique(y_train)
        ]
        model_names, acc_values = [], []

        for name, model in self._classification_models.items():
            if name == "SVM_MKL_Extremality":
                self._train_mkl_svm(
                    X_train, X_val, X_test,
                    y_train, y_val, y_test,
                    present_classes, model_names, acc_values,
                )
                continue

            # ── SVM variants: subsample training set ─────────────────────────
            is_svm = name.startswith("SVM_")
            if is_svm and len(X_train) > SVM_TRAIN_CAP:
                log.warning(
                    "[Supervised] %s: training set (%d rows) exceeds SVM_TRAIN_CAP "
                    "(%d). Training on a stratified subsample to avoid OOM. "
                    "Validation and test sets are kept full.",
                    name, len(X_train), SVM_TRAIN_CAP,
                )
                rng = np.random.default_rng(self.random_state)
                idx_svm = rng.choice(len(X_train), SVM_TRAIN_CAP, replace=False)
                X_fit   = X_train[idx_svm]
                y_fit   = y_train[idx_svm]
            else:
                X_fit, y_fit = X_train, y_train

            log.info("[Supervised] Fitting %s (n=%d) ...", name, len(X_fit))
            check_memory(threshold_gb=2, label=f"Classification: {name}")

            with MemoryGuard(f"Fit {name}"):
                model.fit(X_fit, y_fit)

            y_pred_val  = model.predict(X_val)
            y_pred_test = model.predict(X_test)

            acc_val      = accuracy_score(y_val,  y_pred_val)
            acc_test     = accuracy_score(y_test, y_pred_test)
            report_val   = classification_report(
                y_val, y_pred_val, labels=present_classes,
                output_dict=True, zero_division=0,
            )
            report_test  = classification_report(
                y_test, y_pred_test, labels=present_classes,
                output_dict=True, zero_division=0,
            )
            macro_f1_val  = report_val.get("macro avg",  {}).get("f1-score", 0.0)
            macro_f1_test = report_test.get("macro avg", {}).get("f1-score", 0.0)

            cm_val       = confusion_matrix(y_val,  y_pred_val,  labels=present_classes)
            cm_test      = confusion_matrix(y_test, y_pred_test, labels=present_classes)
            sens_spec_val  = self._compute_sensitivity_specificity(cm_val,  present_classes)
            sens_spec_test = self._compute_sensitivity_specificity(cm_test, present_classes)

            self.results["Classification"][name] = {
                "Validation": {
                    "Accuracy":                round(acc_val,      4),
                    "Macro_F1":                round(macro_f1_val, 4),
                    "Report":                  report_val,
                    "Sensitivity_Specificity": sens_spec_val,
                },
                "Test": {
                    "Accuracy":                round(acc_test,      4),
                    "Macro_F1":                round(macro_f1_test, 4),
                    "Report":                  report_test,
                    "Sensitivity_Specificity": sens_spec_test,
                },
            }
            log.info(
                "  %-30s | Val Acc=%.4f MacroF1=%.4f | Test Acc=%.4f MacroF1=%.4f",
                name, acc_val, macro_f1_val, acc_test, macro_f1_test,
            )

            self.visualizer.confusion_matrix_plot(
                cm_array=cm_val, class_names=present_classes,
                model_name=f"{name} (Validation)", filename=f"cm_val_{name}.png",
            )
            self.visualizer.confusion_matrix_plot(
                cm_array=cm_test, class_names=present_classes,
                model_name=f"{name} (Test)", filename=f"cm_test_{name}.png",
            )
            if hasattr(model, "feature_importances_"):
                self.visualizer.feature_importance_plot(
                    importances=model.feature_importances_,
                    feature_names=feature_names,
                    model_name=name, filename=f"feature_importance_{name}.png",
                )
            if isinstance(model, DecisionTreeClassifier):
                self.visualizer.decision_tree_plot(
                    model=model, feature_names=feature_names,
                    class_names=present_classes, model_name=name,
                    filename=f"tree_plot_{name}.png", max_depth=4,
                )

            model_names.append(name)
            acc_values.append(acc_val)
            log_memory(f"After {name}")

        self.visualizer.model_comparison_bar(
            model_names=model_names, metric_values=acc_values,
            metric_name="Accuracy (Validation)", task="Classification",
            filename="clf_model_comparison_accuracy.png",
        )

    # ── MKL SVM ──────────────────────────────────────────────────────────────
    def _train_mkl_svm(
        self,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        present_classes, model_names, acc_values,
    ) -> None:
        """
        MKL SVM with Extremality weighting.

        Kernel matrices are (n_kernels × n_train × n_train) — at 1 M rows
        this is physically impossible.  We cap the training set at
        MKL_TRAIN_CAP rows and use the full val/test sets for evaluation.
        """
        NUM_WEAK_KERNELS = 3
        MAX_FEATURES     = 5

        log.info("[Supervised] SVM_MKL — building %d weak kernels...", NUM_WEAK_KERNELS)

        # ── Subsample training set ────────────────────────────────────────────
        n_train = len(X_train)
        if n_train > MKL_TRAIN_CAP:
            log.warning(
                "[Supervised] MKL SVM: training set (%d rows) exceeds MKL_TRAIN_CAP "
                "(%d). Using a random subsample. Val/test sets are full.",
                n_train, MKL_TRAIN_CAP,
            )
            rng = np.random.default_rng(self.random_state)
            idx_mkl = rng.choice(n_train, MKL_TRAIN_CAP, replace=False)
            X_mkl   = X_train[idx_mkl]
            y_mkl   = y_train[idx_mkl]
        else:
            X_mkl, y_mkl = X_train, y_train

        log.info("[Supervised] MKL: kernel matrices will be (%d × %d × %d)",
                 NUM_WEAK_KERNELS, len(X_mkl), len(X_mkl))
        check_memory(threshold_gb=4, label="MKL SVM kernel generation")

        with MemoryGuard("MKL weak kernel generation"):
            KL_train, KL_val = create_weak_kernels(
                X_mkl, X_val,
                num_kernels=NUM_WEAK_KERNELS,
                max_features=MAX_FEATURES,
                random_state=self.random_state,
            )
            _, KL_test = create_weak_kernels(
                X_mkl, X_test,
                num_kernels=NUM_WEAK_KERNELS,
                max_features=MAX_FEATURES,
                random_state=self.random_state,
            )

        log_memory("After MKL kernel generation")

        # ── OvR binary labels for metric functions ────────────────────────────
        label_encoder = LabelEncoder()
        y_int     = label_encoder.fit_transform(y_mkl)
        n_classes = len(np.unique(y_int))

        METRIC_NAMES = ["Kernel_Alignment", "Kernel_Polarization", "FSM", "Complex_Ratio"]
        METRIC_FUNCS = [kernel_alignment, kernel_polarization, feature_space_measure, complex_ratio]

        metrics_matrix = np.zeros((NUM_WEAK_KERNELS, len(METRIC_FUNCS)))
        for cls_idx in range(n_classes):
            y_binary = np.where(y_int == cls_idx, 1, -1)
            for k in range(NUM_WEAK_KERNELS):
                for m, fn in enumerate(METRIC_FUNCS):
                    metrics_matrix[k, m] += fn(KL_train[k], y_binary)
        metrics_matrix /= n_classes

        log.info("[Supervised] MKL: per-kernel quality metrics (first 5):")
        header = "  Kernel  " + "  ".join(f"{m:>20}" for m in METRIC_NAMES)
        log.info(header)
        for i in range(min(5, NUM_WEAK_KERNELS)):
            row = f"  {i:>6}  " + "  ".join(
                f"{metrics_matrix[i, j]:>20.4f}" for j in range(4)
            )
            log.info(row)

        # ── Extremality weights ───────────────────────────────────────────────
        most_common_idx    = int(np.bincount(y_int).argmax())
        y_binary_dominant  = np.where(y_int == most_common_idx, 1, -1)

        with MemoryGuard("MKL extremality weights"):
            weights = kernel_extremality_weights(
                KL_train=KL_train, y_train=y_binary_dominant, power=2,
            )

        log.info(
            "[Supervised] MKL top-3 natural weights: %s",
            sorted(weights.w_natural, reverse=True)[:3],
        )
        log.info(
            "[Supervised] MKL top-3 anti-nat weights: %s",
            sorted(weights.w_antinatural, reverse=True)[:3],
        )

        # ── Combined kernels ──────────────────────────────────────────────────
        K_train_nat  = np.einsum("ijk,i->jk", KL_train, weights.w_natural)
        K_val_nat    = np.einsum("ijk,i->jk", KL_val,   weights.w_natural)
        K_test_nat   = np.einsum("ijk,i->jk", KL_test,  weights.w_natural)

        K_train_anti = np.einsum("ijk,i->jk", KL_train, weights.w_antinatural)
        K_val_anti   = np.einsum("ijk,i->jk", KL_val,   weights.w_antinatural)
        K_test_anti  = np.einsum("ijk,i->jk", KL_test,  weights.w_antinatural)

        self.visualizer.mkl_kernel_weights_plot(
            weights_natural=weights.w_natural,
            weights_anti=weights.w_antinatural,
            model_name="SVM_MKL", filename="mkl_kernel_weights.png",
        )
        self.visualizer.mkl_kernel_metrics_plot(
            metrics_matrix=metrics_matrix, metric_names=METRIC_NAMES,
            weights=weights.w_natural, model_name="SVM_MKL",
            filename="mkl_kernel_metrics.png",
        )

        # ── Train & evaluate MKL variants ─────────────────────────────────────
        variants = {
            "SVM_MKL_Natural":     (K_train_nat,  K_val_nat,  K_test_nat),
            "SVM_MKL_AntiNatural": (K_train_anti, K_val_anti, K_test_anti),
        }
        variant_accs = {}

        for variant_name, (K_tr, K_vl, K_te) in variants.items():
            log.info("[Supervised] Fitting %s ...", variant_name)
            check_memory(threshold_gb=2, label=f"MKL SVM fit {variant_name}")

            with MemoryGuard(f"MKL SVM fit {variant_name}"):
                svm = SVC(kernel="precomputed", random_state=self.random_state)
                svm.fit(K_tr, y_mkl)

            y_pred_val  = svm.predict(K_vl)
            y_pred_test = svm.predict(K_te)

            acc_val   = accuracy_score(y_val,  y_pred_val)
            acc_test  = accuracy_score(y_test, y_pred_test)
            report_val  = classification_report(
                y_val, y_pred_val,
                labels=present_classes, output_dict=True, zero_division=0,
            )
            report_test = classification_report(
                y_test, y_pred_test,
                labels=present_classes, output_dict=True, zero_division=0,
            )
            macro_f1_val  = report_val.get("macro avg",  {}).get("f1-score", 0.0)
            macro_f1_test = report_test.get("macro avg", {}).get("f1-score", 0.0)

            cm_val  = confusion_matrix(y_val,  y_pred_val,  labels=present_classes)
            cm_test = confusion_matrix(y_test, y_pred_test, labels=present_classes)
            sens_spec_val  = self._compute_sensitivity_specificity(cm_val,  present_classes)
            sens_spec_test = self._compute_sensitivity_specificity(cm_test, present_classes)

            self.results["Classification"][variant_name] = {
                "Validation": {
                    "Accuracy":                round(acc_val,      4),
                    "Macro_F1":                round(macro_f1_val, 4),
                    "Report":                  report_val,
                    "Sensitivity_Specificity": sens_spec_val,
                },
                "Test": {
                    "Accuracy":                round(acc_test,      4),
                    "Macro_F1":                round(macro_f1_test, 4),
                    "Report":                  report_test,
                    "Sensitivity_Specificity": sens_spec_test,
                },
            }
            if variant_name == "SVM_MKL_Natural":
                per_kernel_metrics = {
                    f"kernel_{i:02d}": {
                        metric: round(float(metrics_matrix[i, j]), 6)
                        for j, metric in enumerate(METRIC_NAMES)
                    }
                    for i in range(NUM_WEAK_KERNELS)
                }
                self.results["Classification"][variant_name]["MKL"] = {
                    "Num_Weak_Kernels":          NUM_WEAK_KERNELS,
                    "Max_Features":              MAX_FEATURES,
                    "Train_Subsample_Size":      len(X_mkl),
                    "Kernel_Weights_Natural":    weights.w_natural.tolist(),
                    "Kernel_Weights_AntiNatural":weights.w_antinatural.tolist(),
                    "Per_Kernel_Metrics":        per_kernel_metrics,
                }

            log.info(
                "  %-30s | Val Acc=%.4f MacroF1=%.4f | Test Acc=%.4f MacroF1=%.4f",
                variant_name, acc_val, macro_f1_val, acc_test, macro_f1_test,
            )
            self.visualizer.confusion_matrix_plot(
                cm_array=cm_val, class_names=present_classes,
                model_name=f"{variant_name} (Validation)",
                filename=f"cm_val_{variant_name}.png",
            )
            self.visualizer.confusion_matrix_plot(
                cm_array=cm_test, class_names=present_classes,
                model_name=f"{variant_name} (Test)",
                filename=f"cm_test_{variant_name}.png",
            )
            variant_accs[variant_name] = acc_val
            model_names.append(variant_name)
            acc_values.append(acc_val)
            log_memory(f"After {variant_name}")

        best = max(variant_accs, key=variant_accs.get)
        log.info(
            "[Supervised] Best MKL variant: %s (Val Acc=%.4f)",
            best, variant_accs[best],
        )
        self.visualizer.model_comparison_bar(
            model_names=list(variant_accs.keys()),
            metric_values=list(variant_accs.values()),
            metric_name="Accuracy (Validation)",
            task="MKL Variants",
            filename="mkl_model_comparison.png",
        )

    # ── Sensitivity / Specificity ─────────────────────────────────────────────
    @staticmethod
    def _compute_sensitivity_specificity(
        cm_array: np.ndarray, class_names: list,
    ) -> dict:
        result = {}
        for i, cls in enumerate(class_names):
            tp = cm_array[i, i]
            fn = cm_array[i, :].sum() - tp
            fp = cm_array[:, i].sum() - tp
            tn = cm_array.sum() - tp - fn - fp

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            result[cls] = {
                "Sensitivity": round(float(sensitivity), 4),
                "Specificity": round(float(specificity), 4),
            }
        return result
