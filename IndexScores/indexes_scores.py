"""
IndexScores/indexes_scores.py — Metrics formatting and export (cloud-optimised).

Changes vs. original: all `print()` replaced with structured logging.
"""

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)


class IndexesScoresEvaluator:
    """Formats and persists all pipeline metrics."""

    def __init__(self, metrics_dict: dict, output_dir: str = "output_metrics") -> None:
        self.metrics    = metrics_dict
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_evaluation(self) -> None:
        self.display_clustering_scores()
        self.display_regression_scores()
        self.display_classification_scores()
        self.display_sensitivity_specificity()
        self.export_metrics_to_json()

    # ── Clustering ────────────────────────────────────────────────────────────
    def display_clustering_scores(self) -> Optional[pd.DataFrame]:
        log.info("=" * 60)
        log.info("CLUSTERING MODELS — EVALUATION SUMMARY")
        log.info("=" * 60)

        clust_data = self.metrics.get("Clustering", {})
        if not clust_data:
            log.warning("No clustering metrics found.")
            return None

        rows = {
            model_name: {
                "N_Clusters":     data.get("N_Clusters"),
                "Silhouette":     data.get("Silhouette"),
                "Davies_Bouldin": data.get("Davies_Bouldin"),
            }
            for model_name, data in clust_data.items()
        }

        df_clust = pd.DataFrame(rows).T.sort_values(by="Silhouette", ascending=False)
        log.info("\n%s", df_clust.to_string())
        return df_clust

    # ── Regression ────────────────────────────────────────────────────────────
    def display_regression_scores(self) -> Optional[pd.DataFrame]:
        log.info("=" * 60)
        log.info("REGRESSION MODELS — LEADERBOARD")
        log.info("=" * 60)

        reg_data = self.metrics.get("Regression", {})
        if not reg_data:
            log.warning("No regression metrics found.")
            return None

        rows = {}
        for model_name, splits in reg_data.items():
            val  = splits.get("Validation", {})
            test = splits.get("Test", {})
            rows[model_name] = {
                "Val_R2":   val.get("R2"),   "Val_RMSE":  val.get("RMSE"),
                "Val_MAE":  val.get("MAE"),  "Val_MSE":   val.get("MSE"),
                "Test_R2":  test.get("R2"),  "Test_RMSE": test.get("RMSE"),
                "Test_MAE": test.get("MAE"), "Test_MSE":  test.get("MSE"),
            }

        df_reg = pd.DataFrame(rows).T.sort_values(by="Val_R2", ascending=False)
        log.info("\n%s", df_reg.to_string())
        return df_reg

    # ── Classification ────────────────────────────────────────────────────────
    def display_classification_scores(self) -> Optional[pd.DataFrame]:
        log.info("=" * 60)
        log.info("CLASSIFICATION MODELS — LEADERBOARD")
        log.info("=" * 60)

        clf_data = self.metrics.get("Classification", {})
        if not clf_data:
            log.warning("No classification metrics found.")
            return None

        rows = {}
        for model_name, splits in clf_data.items():
            val  = splits.get("Validation", {})
            test = splits.get("Test", {})
            rows[model_name] = {
                "Val_Accuracy":  val.get("Accuracy"),
                "Val_MacroF1":   val.get("Macro_F1"),
                "Test_Accuracy": test.get("Accuracy"),
                "Test_MacroF1":  test.get("Macro_F1"),
            }

        df_clf = pd.DataFrame(rows).T.sort_values(by="Val_Accuracy", ascending=False)
        log.info("\n%s", df_clf.to_string())
        return df_clf

    # ── Sensitivity / Specificity ─────────────────────────────────────────────
    def display_sensitivity_specificity(self) -> None:
        log.info("=" * 60)
        log.info("CLASSIFICATION MODELS — SENSITIVITY & SPECIFICITY (per class)")
        log.info("=" * 60)

        clf_data = self.metrics.get("Classification", {})
        if not clf_data:
            log.warning("No classification metrics found.")
            return

        for model_name, splits in clf_data.items():
            log.info("  Model: %s", model_name)

            for split_name in ("Validation", "Test"):
                split_data = splits.get(split_name, {})
                sens_spec  = split_data.get("Sensitivity_Specificity", {})
                if not sens_spec:
                    continue

                lines = [f"    [{split_name}]", f"    {'Class':<22}  {'Sensitivity':>12}  {'Specificity':>12}"]
                lines.append("    " + "-" * 50)

                for cls, values in sens_spec.items():
                    sens = values.get("Sensitivity", float("nan"))
                    spec = values.get("Specificity", float("nan"))
                    lines.append(f"    {cls:<22}  {sens:>12.4f}  {spec:>12.4f}")

                log.info("\n%s", "\n".join(lines))

    # ── JSON export ───────────────────────────────────────────────────────────
    def export_metrics_to_json(self) -> None:
        filepath = os.path.join(self.output_dir, "model_metrics.json")
        clean_metrics = _make_json_serialisable(self.metrics)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(clean_metrics, f, indent=4)

        log.info("[IndexesScores] Full metrics exported to: %s", filepath)


def _make_json_serialisable(obj):
    if isinstance(obj, dict):
        return {k: _make_json_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
