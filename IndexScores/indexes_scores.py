import json
import os
from typing import Optional
import pandas as pd

class IndexesScoresEvaluator:
    """
    Formats and persists all pipeline metrics.
    """

    def __init__(
        self,
        metrics_dict: dict,
        output_dir: str = "output_metrics",
    ) -> None:
        self.metrics = metrics_dict
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # Function to run the full evaluation and export pipeline
    def run_evaluation(self) -> None:
        """Execute the full evaluation and export pipeline."""
        self.display_clustering_scores()
        self.display_regression_scores()
        self.display_classification_scores()
        self.display_sensitivity_specificity()
        self.export_metrics_to_json()

    # Clustering comparison
    def display_clustering_scores(self) -> Optional[pd.DataFrame]:
        """
        Print a formatted leaderboard for all clustering models.

        Metrics displayed: N_Clusters, Silhouette, Davies_Bouldin.
        Sorted by Silhouette Score (descending).
        """

        print("\n" + "=" * 60)
        print("CLUSTERING MODELS — EVALUATION SUMMARY")
        print("=" * 60)

        clust_data = self.metrics.get("Clustering", {})
        if not clust_data:
            print("  No clustering metrics found.")
            return None

        rows = {}
        for model_name, data in clust_data.items():
            rows[model_name] = {
                "N_Clusters":     data.get("N_Clusters"),
                "Silhouette":     data.get("Silhouette"),
                "Davies_Bouldin": data.get("Davies_Bouldin"),
            }

        df_clust = pd.DataFrame(rows).T
        df_clust = df_clust.sort_values(by="Silhouette", ascending=False)

        print(f"\n{df_clust.to_string()}\n")
        return df_clust

    # Regression comparison
    def display_regression_scores(self) -> Optional[pd.DataFrame]:
        """
        Print a formatted leaderboard for all regression models.

        Validation and test metrics are shown side by side.
        Sorted by Validation R2 (descending — higher R2 is better).
        """

        print("\n" + "=" * 60)
        print("REGRESSION MODELS — LEADERBOARD")
        print("=" * 60)

        reg_data = self.metrics.get("Regression", {})
        if not reg_data:
            print("  No regression metrics found.")
            return None

        rows = {}
        for model_name, splits in reg_data.items():
            val  = splits.get("Validation", {})
            test = splits.get("Test", {})
            rows[model_name] = {
                "Val_R2":    val.get("R2"),
                "Val_RMSE":  val.get("RMSE"),
                "Val_MAE":   val.get("MAE"),
                "Val_MSE":   val.get("MSE"),
                "Test_R2":   test.get("R2"),
                "Test_RMSE": test.get("RMSE"),
                "Test_MAE":  test.get("MAE"),
                "Test_MSE":  test.get("MSE"),
            }

        df_reg = pd.DataFrame(rows).T
        df_reg = df_reg.sort_values(by="Val_R2", ascending=False)

        print(f"\n{df_reg.to_string()}\n")
        return df_reg

    # Classification comparison
    def display_classification_scores(self) -> Optional[pd.DataFrame]:
        """
        Print a formatted leaderboard for all classification models.

        Summary columns: Accuracy and Macro F1-Score for both validation
        and test sets.  Sorted by Validation Accuracy (descending).
        """

        print("\n" + "=" * 60)
        print("CLASSIFICATION MODELS — LEADERBOARD")
        print("=" * 60)

        clf_data = self.metrics.get("Classification", {})
        if not clf_data:
            print("  No classification metrics found.")
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

        df_clf = pd.DataFrame(rows).T
        df_clf = df_clf.sort_values(by="Val_Accuracy", ascending=False)

        print(f"\n{df_clf.to_string()}\n")
        return df_clf

    # Sensitivity / Specificity detail table
    def display_sensitivity_specificity(self) -> None:
        """
        Print per-class Sensitivity and Specificity for every classification
        model, for both the validation and test sets.

        These are derived from the one-vs-rest decomposition of the
        confusion matrix stored during training.

        Interpretation:
            Sensitivity (Recall / TPR) — proportion of true cases of class C
                correctly identified.  Critical when missing a true positive
                (false negative) carries a high cost, e.g., failing to detect
                a severe depressive episode.
            Specificity (TNR) — proportion of true non-cases correctly
                excluded.  Critical when a false alarm (false positive) carries
                a high cost.
        """

        print("\n" + "=" * 60)
        print("CLASSIFICATION MODELS — SENSITIVITY & SPECIFICITY (per class)")
        print("=" * 60)

        clf_data = self.metrics.get("Classification", {})
        if not clf_data:
            print("  No classification metrics found.")
            return

        for model_name, splits in clf_data.items():
            print(f"\n  Model: {model_name}")

            for split_name in ("Validation", "Test"):
                split_data = splits.get(split_name, {})
                sens_spec  = split_data.get("Sensitivity_Specificity", {})

                if not sens_spec:
                    continue

                print(f"    [{split_name}]")
                header = f"    {'Class':<22s}  {'Sensitivity':>12}  {'Specificity':>12}"
                print(header)
                print("    " + "-" * (len(header) - 4))

                for cls, values in sens_spec.items():
                    sens = values.get("Sensitivity", float("nan"))
                    spec = values.get("Specificity", float("nan"))
                    print(
                        f"    {cls:<22s}  {sens:>12.4f}  {spec:>12.4f}"
                    )

    # JSON export
    def export_metrics_to_json(self) -> None:
        """
        Serialise the full metrics dictionary to a JSON file.

        The file is written to ``output_dir/model_metrics.json``.
        All numeric values are stored with full floating-point precision.
        """
        filepath = os.path.join(self.output_dir, "model_metrics.json")

        # Convert any non-serialisable values (e.g. numpy floats) before dumping
        clean_metrics = _make_json_serialisable(self.metrics)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(clean_metrics, f, indent=4)

        print(f"\n[IndexesScores] Full metrics exported to: {filepath}")


# Utility: recursive JSON sanitiser
def _make_json_serialisable(obj):
    """
    Recursively convert numpy scalar types to native Python types so that
    ``json.dump`` does not raise a TypeError.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_json_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
