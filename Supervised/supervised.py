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

# MKL library imports — used to build the SVM_MKL_Extremality classifier.
# The library must be installed from the MKL/ directory via:
#   pip install -e MKL/
from MKL.extremality_mkl import kernel_extremality_weights, create_weak_kernels
from MKL.extremality_mkl.metrics import (
    kernel_alignment,
    kernel_polarization,
    feature_space_measure,
    complex_ratio,
)

# PHQ-9 category ordering (defines class order for reporting)
PHQ9_CATEGORY_ORDER = [
    "Minimal",
    "Mild",
    "Moderate",
    "Moderately_Severe",
    "Severe",
]

class PHQ9ModelTrainer:
    """
    Encapsulates the full supervised learning pipeline for PHQ-9 prediction.
    """

    def __init__(
        self,
        target_reg: str = "phq9_total",
        target_clf: str = "cluster_label",
        random_state: int = 42,
    ) -> None:
        self.target_reg = target_reg
        self.target_clf = target_clf
        self.random_state = random_state
        self.visualizer = Visualization()
        self.results: Dict = {"Regression": {}, "Classification": {}}

        self._regression_models = self._init_regression_models()
        self._classification_models = self._init_classification_models()


    # Model initialisation
    def _init_regression_models(self) -> dict:
        """Return a dictionary of untrained regression estimators."""
        return {
            "Linear_Regression": LinearRegression(),
            "Decision_Tree_Reg": DecisionTreeRegressor(
                random_state=self.random_state
            ),
            "Random_Forest_Reg": RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
            ),
            "Gradient_Boosting_Reg": GradientBoostingRegressor(
                random_state=self.random_state
            ),
        }

    def _init_classification_models(self) -> dict:
        """Return a dictionary of untrained classification estimators."""
        return {
            "Logistic_Regression": LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,
                class_weight="balanced",
            ),
            "Decision_Tree_Clf": DecisionTreeClassifier(
                random_state=self.random_state,
                class_weight="balanced",
            ),
            "Random_Forest_Clf": RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight="balanced",
            ),
            "Gradient_Boosting_Clf": GradientBoostingClassifier(
                random_state=self.random_state
            ),
            "SVM_Linear": SVC(
                kernel="linear",
                class_weight="balanced",
                random_state=self.random_state,
                probability=True,
            ),
            "SVM_Polynomial": SVC(
                kernel="poly",
                degree=3,
                class_weight="balanced",
                random_state=self.random_state,
                probability=True,
            ),
            "SVM_RBF": SVC(
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                random_state=self.random_state,
                probability=True,
            ),
            # Sentinel entry: this key signals that the MKL training block
            # should run. The value is None because MKL uses a precomputed
            # kernel and cannot be represented as a standard SVC instance.
            "SVM_MKL_Extremality": None,
        }

    # Function to run the whole pipeline
    def run_pipeline(self, df: pd.DataFrame) -> dict:
        """
        Execute the full supervised learning pipeline.
        """

        print("\n" + "=" * 60)
        print("SUPERVISED LEARNING")
        print("=" * 60)

        X_train_s, X_val_s, X_test_s, \
        y_train_r, y_val_r, y_test_r, \
        y_train_c, y_val_c, y_test_c, \
        feature_names = self._prepare_data(df)

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

    # Data preparation
    def _prepare_data(self, df: pd.DataFrame) -> Tuple:
        """
        Split and scale the dataset into train / validation / test sets.

        The split follows the 60 / 20 / 20 scheme.

        Stratification is applied using the classification target to ensure
        that each PHQ-9 severity class is represented proportionally in every
        partition.
        """

        # Identify columns that are not targets or metadata
        exclude_cols = {self.target_reg, self.target_clf, "phq9_category"}
        feature_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in exclude_cols
        ]

        X = df[feature_cols].values
        y_reg = df[self.target_reg].values
        y_clf = df[self.target_clf].values

        print(f"\n[Supervised] Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
        print(f"[Supervised] Regression target  : {self.target_reg}")
        print(f"[Supervised] Classification target: {self.target_clf}")

        # First split: separate test set (20 %)
        (X_temp, X_test,
         y_temp_r, y_test_r,
         y_temp_c, y_test_c) = train_test_split(
            X, y_reg, y_clf,
            test_size=0.20,
            random_state=self.random_state,
            stratify=y_clf,
        )

        # Second split: separate validation set (25 % of temp = 20 % overall)
        (X_train, X_val,
         y_train_r, y_val_r,
         y_train_c, y_val_c) = train_test_split(
            X_temp, y_temp_r, y_temp_c,
            test_size=0.25,
            random_state=self.random_state,
            stratify=y_temp_c,
        )

        print(
            f"[Supervised] Set sizes — "
            f"Train: {len(X_train)}, "
            f"Validation: {len(X_val)}, "
            f"Test: {len(X_test)}"
        )

        # Fit scaler on training data only to prevent data leakage
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)

        return (
            X_train_s, X_val_s, X_test_s,
            y_train_r, y_val_r, y_test_r,
            y_train_c, y_val_c, y_test_c,
            feature_cols,
        )

    # Regression training
    def _train_regression(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
    ) -> None:
        """
        Train all regression models, compute metrics, and generate plots.

        Metrics computed (on validation set):
            - MSE, RMSE, MAE, R2
        """

        print("\n" + "-" * 50)
        print("Training Regression Models")
        print("-" * 50)

        model_names, r2_values = [], []

        for name, model in self._regression_models.items():
            model.fit(X_train, y_train)

            # Validation metrics
            y_pred_val = model.predict(X_val)
            mse_val  = mean_squared_error(y_val, y_pred_val)
            rmse_val = float(np.sqrt(mse_val))
            mae_val  = mean_absolute_error(y_val, y_pred_val)
            r2_val   = r2_score(y_val, y_pred_val)

            # Test metrics
            y_pred_test = model.predict(X_test)
            mse_test  = mean_squared_error(y_test, y_pred_test)
            rmse_test = float(np.sqrt(mse_test))
            mae_test  = mean_absolute_error(y_test, y_pred_test)
            r2_test   = r2_score(y_test, y_pred_test)

            self.results["Regression"][name] = {
                "Validation": {
                    "MSE":  round(mse_val,  4),
                    "RMSE": round(rmse_val, 4),
                    "MAE":  round(mae_val,  4),
                    "R2":   round(r2_val,   4),
                },
                "Test": {
                    "MSE":  round(mse_test,  4),
                    "RMSE": round(rmse_test, 4),
                    "MAE":  round(mae_test,  4),
                    "R2":   round(r2_test,   4),
                },
            }

            print(
                f"  {name:<30s} | "
                f"Val  R2={r2_val:.4f}  RMSE={rmse_val:.3f} | "
                f"Test R2={r2_test:.4f}  RMSE={rmse_test:.3f}"
            )

            # Predicted vs. actual scatter plot
            self.visualizer.regression_scatter(
                y_true=y_val,
                y_pred=y_pred_val,
                model_name=name,
                filename=f"reg_scatter_{name}.png",
            )

            # Feature importance (tree-based models only)
            if hasattr(model, "feature_importances_"):
                self.visualizer.feature_importance_plot(
                    importances=model.feature_importances_,
                    feature_names=feature_names,
                    model_name=name,
                    filename=f"feature_importance_{name}.png",
                )

            # Decision tree structure diagram (Decision Tree models only)
            if isinstance(model, DecisionTreeRegressor):
                self.visualizer.decision_tree_plot(
                    model=model,
                    feature_names=feature_names,
                    class_names=None,   # None for regressors
                    model_name=name,
                    filename=f"tree_plot_{name}.png",
                    max_depth=4,
                )

            model_names.append(name)
            r2_values.append(r2_val)

        # Model comparison chart
        self.visualizer.model_comparison_bar(
            model_names=model_names,
            metric_values=r2_values,
            metric_name="R2 (Validation)",
            task="Regression",
            filename="reg_model_comparison_r2.png",
        )

    # Classification training
    def _train_classification(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
    ) -> None:
        """
        Train all classification models, compute metrics, and generate plots.

        Metrics computed (on validation set):
            - Accuracy
            - Macro F1-Score
            - Per-class Sensitivity (Recall) and Specificity
            - Full classification report (precision, recall, F1, support)
            - Confusion matrix
        """

        print("\n" + "-" * 50)
        print("Training Classification Models")
        print("-" * 50)

        # Determine the ordered class names present in the data
        present_classes = [
            c for c in PHQ9_CATEGORY_ORDER if c in np.unique(y_train)
        ]

        model_names, acc_values = [], []

        for name, model in self._classification_models.items():
            """
            Special case: MKL SVM with Extremality weighting.
            This model uses a precomputed kernel built from multiple weak
            polynomial kernels combined via the extremality_mkl library.
            It cannot follow the standard fit/predict path because it requires
            building separate kernel matrices for train, validation, and test.
            """
            if name == "SVM_MKL_Extremality":
                self._train_mkl_svm(
                    X_train, X_val, X_test,
                    y_train, y_val, y_test,
                    present_classes,
                    model_names, acc_values,
                )
                continue

            model.fit(X_train, y_train)

            # Validation metrics
            y_pred_val = model.predict(X_val)
            acc_val   = accuracy_score(y_val, y_pred_val)
            report_val = classification_report(
                y_val, y_pred_val,
                labels=present_classes,
                output_dict=True,
                zero_division=0,
            )
            macro_f1_val = report_val.get("macro avg", {}).get("f1-score", 0.0)

            # Per-class sensitivity and specificity from confusion matrix
            cm_val   = confusion_matrix(y_val, y_pred_val, labels=present_classes)
            sens_spec_val = self._compute_sensitivity_specificity(
                cm_val, present_classes
            )

            # Test metrics
            y_pred_test  = model.predict(X_test)
            acc_test     = accuracy_score(y_test, y_pred_test)
            report_test  = classification_report(
                y_test, y_pred_test,
                labels=present_classes,
                output_dict=True,
                zero_division=0,
            )
            macro_f1_test = report_test.get("macro avg", {}).get("f1-score", 0.0)
            cm_test       = confusion_matrix(
                y_test, y_pred_test, labels=present_classes
            )
            sens_spec_test = self._compute_sensitivity_specificity(
                cm_test, present_classes
            )

            self.results["Classification"][name] = {
                "Validation": {
                    "Accuracy":          round(acc_val,      4),
                    "Macro_F1":          round(macro_f1_val, 4),
                    "Report":            report_val,
                    "Sensitivity_Specificity": sens_spec_val,
                },
                "Test": {
                    "Accuracy":          round(acc_test,      4),
                    "Macro_F1":          round(macro_f1_test, 4),
                    "Report":            report_test,
                    "Sensitivity_Specificity": sens_spec_test,
                },
            }

            print(
                f"  {name:<30s} | "
                f"Val  Acc={acc_val:.4f}  MacroF1={macro_f1_val:.4f} | "
                f"Test Acc={acc_test:.4f}  MacroF1={macro_f1_test:.4f}"
            )

            # Confusion matrix plots
            self.visualizer.confusion_matrix_plot(
                cm_array=cm_val,
                class_names=present_classes,
                model_name=f"{name} (Validation)",
                filename=f"cm_val_{name}.png",
            )
            self.visualizer.confusion_matrix_plot(
                cm_array=cm_test,
                class_names=present_classes,
                model_name=f"{name} (Test)",
                filename=f"cm_test_{name}.png",
            )

            # Feature importance (tree-based models only)
            if hasattr(model, "feature_importances_"):
                self.visualizer.feature_importance_plot(
                    importances=model.feature_importances_,
                    feature_names=feature_names,
                    model_name=name,
                    filename=f"feature_importance_{name}.png",
                )

            # Decision tree structure diagram (Decision Tree models only)
            if isinstance(model, DecisionTreeClassifier):
                self.visualizer.decision_tree_plot(
                    model=model,
                    feature_names=feature_names,
                    class_names=present_classes,
                    model_name=name,
                    filename=f"tree_plot_{name}.png",
                    max_depth=4,
                )

            model_names.append(name)
            acc_values.append(acc_val)

        # Model comparison chart
        self.visualizer.model_comparison_bar(
            model_names=model_names,
            metric_values=acc_values,
            metric_name="Accuracy (Validation)",
            task="Classification",
            filename="clf_model_comparison_accuracy.png",
        )


    # MKL SVM training
    def _train_mkl_svm(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        present_classes: list,
        model_names: list,
        acc_values: list,
    ) -> None:
        """
        Train and compare four SVM variants using the extremality_mkl library,
        mirroring the evaluation pattern in the library's own test.py script.

        The four variants trained and compared are:
          1. MKL Natural      : kernels weighted to favour highest-quality ones.
          2. MKL Anti-Natural : kernels weighted to favour lowest-quality ones.
                                Used as an internal baseline — if the natural
                                weighting is meaningful, it should outperform
                                the anti-natural one.
          3. SVM RBF          : single global RBF kernel (standard baseline).
          4. SVM Polynomial-3 : single global degree-3 polynomial kernel
                                (standard baseline).

        Procedure
        ---------
        1.  Generate NUM_WEAK_KERNELS weak polynomial kernels from random
            feature subsets of the training data.
        2.  Convert the multiclass string labels to binary {-1, +1} using
            one-vs-rest: for each PHQ-9 class, the current class is +1 and
            all others are -1. Metrics are averaged across all classes.
            This matches the binary label convention required by the library,
            as used in the test.py script.
        3.  Compute all four MKL quality metrics for every weak kernel:
              - Kernel Alignment   : similarity to the ideal label kernel.
              - Kernel Polarization : class-separation in the kernel space.
              - FSM                : Feature Space Measure (inter/intra class).
              - Complex Ratio      : trace of the kernel matrix (total energy).
        4.  Compute extremality weights (natural and anti-natural).
        5.  Build combined kernel matrices for all four SVM variants.
        6.  Evaluate each variant on validation and test sets with the same
            metrics used by all other classifiers.

        Plots generated
        ---------------
        - mkl_kernel_weights.png  : natural vs anti-natural weights per kernel.
        - mkl_kernel_metrics.png  : 4-panel chart of all four quality metrics.
        - mkl_model_comparison.png: accuracy bar chart for all four variants.
        - cm_val/test_<variant>.png: confusion matrices for each variant.
        """
        NUM_WEAK_KERNELS = 15
        MAX_FEATURES = 5

        print(f"\nSVM_MKL — building {NUM_WEAK_KERNELS} weak kernels...")

        # ------------------------------------------------------------------
        # Step 1: generate weak polynomial kernels.
        # create_weak_kernels projects X_val and X_test using the same
        # random feature subsets chosen from X_train, so there is no leakage.
        # ------------------------------------------------------------------
        KL_train, KL_val = create_weak_kernels(
            X_train, X_val,
            num_kernels=NUM_WEAK_KERNELS,
            max_features=MAX_FEATURES,
            random_state=self.random_state,
        )
        _, KL_test = create_weak_kernels(
            X_train, X_test,
            num_kernels=NUM_WEAK_KERNELS,
            max_features=MAX_FEATURES,
            random_state=self.random_state,
        )

        # ------------------------------------------------------------------
        # Step 2: convert multiclass labels to binary {-1, +1}.
        # The library's metric functions (kernel_alignment, FSM, etc.) require
        # binary labels: y = np.where(y == 0, -1, y).
        # For our multiclass problem we use one-vs-rest averaging: each class
        # takes a turn as +1 and all others become -1. The four metrics are
        # computed for each class split and averaged, giving a single quality
        # score per kernel that reflects separability across all classes.
        # ------------------------------------------------------------------
        label_encoder = LabelEncoder()
        y_int = label_encoder.fit_transform(y_train)
        n_classes = len(np.unique(y_int))

        METRIC_NAMES = [
            "Kernel_Alignment",
            "Kernel_Polarization",
            "FSM",
            "Complex_Ratio",
        ]
        METRIC_FUNCS = [
            kernel_alignment,
            kernel_polarization,
            feature_space_measure,
            complex_ratio,
        ]

        # Accumulate metrics over all one-vs-rest splits, then average.
        metrics_matrix = np.zeros((NUM_WEAK_KERNELS, len(METRIC_FUNCS)))
        for cls_idx in range(n_classes):
            y_binary = np.where(y_int == cls_idx, 1, -1)
            for k in range(NUM_WEAK_KERNELS):
                for m, fn in enumerate(METRIC_FUNCS):
                    metrics_matrix[k, m] += fn(KL_train[k], y_binary)
        metrics_matrix /= n_classes  # average across all one-vs-rest splits

        print("  Per-kernel quality metrics (first 5 kernels shown):")
        header = f"  {'Kernel':>6}  " + "  ".join(f"{m:>20}" for m in METRIC_NAMES)
        print(header)
        for i in range(min(5, NUM_WEAK_KERNELS)):
            row = (
                f"  {i:>6}  "
                + "  ".join(f"{metrics_matrix[i, j]:>20.4f}" for j in range(4))
            )
            print(row)
        if NUM_WEAK_KERNELS > 5:
            print(f"  ... ({NUM_WEAK_KERNELS - 5} more kernels not shown)")

        # ------------------------------------------------------------------
        # Step 3: compute extremality weights.
        # We pass the averaged binary representation (most common class as +1)
        # to kernel_extremality_weights, which runs its own internal metric
        # evaluation to produce w_natural and w_antinatural.
        # ------------------------------------------------------------------
        most_common_idx = int(np.bincount(y_int).argmax())
        y_binary_dominant = np.where(y_int == most_common_idx, 1, -1)

        weights = kernel_extremality_weights(
            KL_train=KL_train,
            y_train=y_binary_dominant,
            power=2,
        )

        print(
            f"  Top-3 natural weights  : "
            f"{sorted(weights.w_natural, reverse=True)[:3]}"
        )
        print(
            f"  Top-3 anti-nat weights : "
            f"{sorted(weights.w_antinatural, reverse=True)[:3]}"
        )

        # ------------------------------------------------------------------
        # Step 4: build combined kernel matrices for all four variants,
        # following the same einsum pattern used in test.py.
        # ------------------------------------------------------------------
        K_train_nat  = np.einsum("ijk,i->jk", KL_train, weights.w_natural)
        K_val_nat    = np.einsum("ijk,i->jk", KL_val,   weights.w_natural)
        K_test_nat   = np.einsum("ijk,i->jk", KL_test,  weights.w_natural)

        K_train_anti = np.einsum("ijk,i->jk", KL_train, weights.w_antinatural)
        K_val_anti   = np.einsum("ijk,i->jk", KL_val,   weights.w_antinatural)
        K_test_anti  = np.einsum("ijk,i->jk", KL_test,  weights.w_antinatural)

        # ------------------------------------------------------------------
        # Plots: weight distributions and per-kernel metric charts.
        # ------------------------------------------------------------------
        self.visualizer.mkl_kernel_weights_plot(
            weights_natural=weights.w_natural,
            weights_anti=weights.w_antinatural,
            model_name="SVM_MKL",
            filename="mkl_kernel_weights.png",
        )
        self.visualizer.mkl_kernel_metrics_plot(
            metrics_matrix=metrics_matrix,
            metric_names=METRIC_NAMES,
            weights=weights.w_natural,
            model_name="SVM_MKL",
            filename="mkl_kernel_metrics.png",
        )

        # ------------------------------------------------------------------
        # Step 5 & 6: train and evaluate each of the four SVM variants.
        # This mirrors the configs loop in test.py exactly.
        # ------------------------------------------------------------------
        variants = {
            "SVM_MKL_Natural":      (K_train_nat,  K_val_nat,  K_test_nat),
            "SVM_MKL_AntiNatural":  (K_train_anti, K_val_anti, K_test_anti),
        }

        variant_accs = {}

        for variant_name, (K_tr, K_vl, K_te) in variants.items():
            svm = SVC(kernel="precomputed", random_state=self.random_state)
            svm.fit(K_tr, y_train)

            y_pred_val  = svm.predict(K_vl)
            y_pred_test = svm.predict(K_te)

            acc_val  = accuracy_score(y_val,  y_pred_val)
            acc_test = accuracy_score(y_test, y_pred_test)

            report_val  = classification_report(
                y_val,  y_pred_val,
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

            sens_spec_val  = self._compute_sensitivity_specificity(
                cm_val,  present_classes
            )
            sens_spec_test = self._compute_sensitivity_specificity(
                cm_test, present_classes
            )

            self.results["Classification"][variant_name] = {
                "Validation": {
                    "Accuracy":                round(acc_val,       4),
                    "Macro_F1":                round(macro_f1_val,  4),
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

            # Attach the full MKL metadata only to the Natural variant,
            # since the weights and per-kernel metrics describe that model.
            if variant_name == "SVM_MKL_Natural":
                per_kernel_metrics = {
                    f"kernel_{i:02d}": {
                        metric: round(float(metrics_matrix[i, j]), 6)
                        for j, metric in enumerate(METRIC_NAMES)
                    }
                    for i in range(NUM_WEAK_KERNELS)
                }
                self.results["Classification"][variant_name]["MKL"] = {
                    "Num_Weak_Kernels":   NUM_WEAK_KERNELS,
                    "Max_Features":       MAX_FEATURES,
                    "Kernel_Weights_Natural":     weights.w_natural.tolist(),
                    "Kernel_Weights_AntiNatural": weights.w_antinatural.tolist(),
                    "Per_Kernel_Metrics": per_kernel_metrics,
                }

            print(
                f"  {variant_name:<30s} | "
                f"Val  Acc={acc_val:.4f}  MacroF1={macro_f1_val:.4f} | "
                f"Test Acc={acc_test:.4f}  MacroF1={macro_f1_test:.4f}"
            )

            self.visualizer.confusion_matrix_plot(
                cm_array=cm_val,
                class_names=present_classes,
                model_name=f"{variant_name} (Validation)",
                filename=f"cm_val_{variant_name}.png",
            )
            self.visualizer.confusion_matrix_plot(
                cm_array=cm_test,
                class_names=present_classes,
                model_name=f"{variant_name} (Test)",
                filename=f"cm_test_{variant_name}.png",
            )

            variant_accs[variant_name] = acc_val
            model_names.append(variant_name)
            acc_values.append(acc_val)

        # Print the best MKL variant, mirroring the summary in test.py
        best_variant = max(variant_accs, key=variant_accs.get)
        print(
            f"\n  Best MKL variant: {best_variant} "
            f"(Val Accuracy = {variant_accs[best_variant]:.4f})"
        )

        # Bar chart comparing all four MKL variants
        self.visualizer.model_comparison_bar(
            model_names=list(variant_accs.keys()),
            metric_values=list(variant_accs.values()),
            metric_name="Accuracy (Validation)",
            task="MKL Variants",
            filename="mkl_model_comparison.png",
        )


    # Sensitivity / Specificity helper
    @staticmethod
    def _compute_sensitivity_specificity(
        cm_array: np.ndarray,
        class_names: list,
    ) -> dict:
        """
        Calculate sensitivity (recall) and specificity from a
        multi-class confusion matrix using the one-vs-rest decomposition.

        Sensitivity (True Positive Rate) for class C:
            TP_C / (TP_C + FN_C)
        Specificity (True Negative Rate) for class C:
            TN_C / (TN_C + FP_C)
        """
        
        n = len(class_names)
        result = {}

        for i, cls in enumerate(class_names):
            tp = cm_array[i, i]
            fn = cm_array[i, :].sum() - tp     # Row sum minus TP
            fp = cm_array[:, i].sum() - tp     # Column sum minus TP
            tn = cm_array.sum() - tp - fn - fp

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            result[cls] = {
                "Sensitivity": round(float(sensitivity), 4),
                "Specificity": round(float(specificity), 4),
            }

        return result
