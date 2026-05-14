import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import umap
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

# Shared style constants
PALETTE_MAIN = "viridis"
PALETTE_PASTEL = sns.color_palette("pastel")
DPI = 150 # Resolution for saved figures

class Visualization:
    """
    Collection of static-style plotting helpers used throughout the pipeline.

    All methods save figures to subdirectories of ``output_dir``
    (default: ``graficos/``) and return ``None``.
    """

    def __init__(self, output_dir: str = "graphs") -> None:
        self.output_dir = Path(output_dir)
        # Create the required sub-directories once at construction time
        for sub in ("eda", "unsupervised", "supervised"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    # Helper function to save and close figures
    def _save(self, fig: plt.Figure, subdir: str, filename: str) -> None:
        """Save *fig* to ``output_dir/subdir/filename`` and close it."""
        path = self.output_dir / subdir / filename
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # EDA Visualizations
    def histogram(self, series: pd.Series, col_name: str, filename: str) -> None:
        """
        Plot a histogram with mean and median reference lines.
        The number of bins is determined by Sturges' rule: k = 1 + log2(n).
        """
        n = len(series)
        bins = max(5, int(1 + np.log2(n)))   # Sturges' rule, minimum 5 bins

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(series, bins=bins, color="steelblue", ax=ax, edgecolor="white")

        mean_val = series.mean()
        median_val = series.median()

        ax.axvline(mean_val,   color="crimson",     linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color="forestgreen", linestyle="-",  linewidth=1.5,
                   label=f"Median: {median_val:.2f}")

        ax.set_title(f"Distribution of {col_name}", fontsize=14)
        ax.set_xlabel(col_name, fontsize=11)
        ax.set_ylabel("Frequency",  fontsize=11)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        self._save(fig, "eda", filename)

    def boxplot(self, series: pd.Series, col_name: str, filename: str) -> None:
        """
        Plot a single boxplot for outlier inspection.
        """
        fig, ax = plt.subplots(figsize=(5, 6))
        sns.boxplot(y=series, color="steelblue", ax=ax, width=0.4)
        ax.set_title(f"Boxplot of {col_name}", fontsize=13)
        ax.set_ylabel(col_name, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        self._save(fig, "eda", filename)

    def pie_chart(self, series: pd.Series, col_name: str, filename: str) -> None:
        """
        Plot a pie chart for a categorical variable.
        Only called when the number of unique categories is small
        """

        counts = series.value_counts()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("pastel", len(counts)),
        )
        ax.set_title(f"Distribution of {col_name}", fontsize=14)
        ax.axis("equal")
        self._save(fig, "eda", filename)

    def correlation_matrix(self, df: pd.DataFrame, filename: str = "correlation_matrix.png") -> None:
        """
        Plot a Pearson correlation heat-map for all numeric columns.
        """

        corr = df.corr(method="pearson")
        mask = np.triu(np.ones_like(corr, dtype=bool))   # Show lower triangle only

        fig, ax = plt.subplots(figsize=(max(10, len(corr) // 2), max(8, len(corr) // 2)))
        sns.heatmap(
            corr,
            mask=mask,
            annot=False,
            cmap="coolwarm",
            center=0,
            linewidths=0.3,
            ax=ax,
            cbar_kws={"shrink": 0.7},
        )
        ax.set_title("Pearson Correlation Matrix", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        self._save(fig, "eda", filename)

    def scatter_vs_target(self, df: pd.DataFrame, feature_cols: list, target_col: str, filename: str = "scatter_vs_target.png") -> None:
        """
        Plot scatter plots of each feature against the target variable
        and annotate with the Pearson correlation coefficient.
        """

        n_cols = 3
        n_rows = int(np.ceil(len(feature_cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 5, n_rows * 4))
        axes = np.array(axes).flatten()

        for i, col in enumerate(feature_cols):
            r = df[col].corr(df[target_col])
            axes[i].scatter(df[col], df[target_col],
                            alpha=0.25, s=10, color="steelblue")
            axes[i].set_xlabel(col,        fontsize=9)
            axes[i].set_ylabel(target_col, fontsize=9)
            axes[i].set_title(f"r = {r:.3f}", fontsize=10)
            axes[i].grid(linestyle="--", alpha=0.4)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Feature Scatter Plots vs {target_col}", fontsize=14, y=1.01)
        fig.tight_layout()
        self._save(fig, "eda", filename)

    def umap_projection(self, df: pd.DataFrame, features_scaled: np.ndarray, target_col: str, filename: str = "umap.png") -> None:
        """
        Compute and plot a 3-D UMAP projection coloured by the target.
        """
        reducer = umap.UMAP(
            n_neighbors=10,
            min_dist=0.3,
            n_components=2,
            low_memory=True,
            random_state=42,
            transform_seed=42,
        )

        # Dimensionality reduction with PCA
        pca = PCA(n_components=15, random_state=42)
        features_pca = pca.fit_transform(features_scaled)

        embedding = reducer.fit_transform(features_pca)

        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection="3d")

        target_vals = df[target_col].values
        sc = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=target_vals,
            cmap=PALETTE_MAIN,
            s=1,
            alpha=0.3,
            rasterized=True,
        )
        fig.colorbar(sc, ax=ax, label=target_col, shrink=0.6)
        ax.set_title("UMAP 3-D Projection", fontsize=13)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_zlabel("UMAP-3")
        self._save(fig, "eda", filename)

    # Unsupervised Visualizations
    def elbow_plot(self, k_values: list, inertias: list, filename: str = "elbow_plot.png") -> None:
        """
        Plot the K-Means elbow curve (inertia vs. number of clusters).
        """

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_values, inertias, marker="o", color="steelblue", linewidth=2)
        ax.set_title("K-Means Elbow Curve", fontsize=13)
        ax.set_xlabel("Number of Clusters (k)", fontsize=11)
        ax.set_ylabel("Inertia (WCSS)",         fontsize=11)
        ax.grid(linestyle="--", alpha=0.5)
        self._save(fig, "unsupervised", filename)

    def silhouette_plot(self, k_values: list, silhouettes: list, filename: str = "silhouette_scores.png") -> None:
        """
        Plot silhouette scores across different values of k.
        """

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_values, silhouettes, marker="s", color="darkorange", linewidth=2)
        ax.set_title("Silhouette Score vs. Number of Clusters", fontsize=13)
        ax.set_xlabel("Number of Clusters (k)", fontsize=11)
        ax.set_ylabel("Mean Silhouette Score", fontsize=11)
        ax.grid(linestyle="--", alpha=0.5)
        self._save(fig, "unsupervised", filename)

    def cluster_scatter_2d(self, embedding_2d: np.ndarray, labels: np.ndarray, title: str, filename: str) -> None:
        """
        Scatter plot of a 2-D embedding coloured by cluster labels.
        Used to visualise any clustering result on a 2-D PCA/UMAP projection.
        """

        unique_labels = np.unique(labels)
        palette = cm.get_cmap("tab10", len(unique_labels))

        fig, ax = plt.subplots(figsize=(9, 7))
        for idx, lbl in enumerate(unique_labels):
            mask = labels == lbl
            label_name = f"Cluster {lbl}" if lbl >= 0 else "Noise"
            ax.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=[palette(idx)],
                label=label_name,
                s=10,
                alpha=0.6,
            )
        ax.set_title(title,   fontsize=13)
        ax.set_xlabel("PC-1", fontsize=11)
        ax.set_ylabel("PC-2", fontsize=11)
        ax.legend(markerscale=2, fontsize=9, loc="best")
        ax.grid(linestyle="--", alpha=0.4)
        self._save(fig, "unsupervised", filename)

    # Supervised Visualizations
    def confusion_matrix_plot(self, cm_array: np.ndarray, class_names: list, model_name: str, filename: str) -> None:
        """
        Plot a labelled confusion matrix heat-map.
        """

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label",      fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        self._save(fig, "supervised", filename)

    def regression_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, filename: str) -> None:
        """
        Plot predicted vs. actual values for a regression model.
        A perfect model would place all points on the diagonal reference line.
        """

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_true, y_pred, alpha=0.35, s=12, color="steelblue")

        # Perfect-prediction diagonal
        lims = [min(y_true.min(), y_pred.min()),
                max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")

        ax.set_title(f"Predicted vs. Actual — {model_name}", fontsize=13)
        ax.set_xlabel("Actual Values",    fontsize=11)
        ax.set_ylabel("Predicted Values", fontsize=11)
        ax.legend()
        ax.grid(linestyle="--", alpha=0.4)
        self._save(fig, "supervised", filename)

    def decision_tree_plot(self, model, feature_names: list, class_names: list, model_name: str, filename: str, max_depth: int = 4) -> None:
        """
        Plot the branching structure of a trained DecisionTreeClassifier or
        DecisionTreeRegressor using sklearn.tree.plot_tree.

        Only the first max_depth levels are rendered to keep the diagram
        readable. Each node shows the splitting feature and threshold, the
        impurity (Gini for classifiers, MSE for regressors), the number of
        training samples reaching that node, and the predicted value or class.

        Parameters
        ----------
        model : DecisionTreeClassifier or DecisionTreeRegressor
            A fitted scikit-learn decision tree estimator.
        feature_names : list of str
            Names of the input features, in the same order as the training
            feature matrix columns.
        class_names : list of str or None
            Ordered class labels for classifiers. Pass None for regressors.
        model_name : str
            Used in the figure title.
        filename : str
            Output filename, saved under graphs/supervised/.
        max_depth : int
            Maximum number of tree levels to display. Default is 4.
        """
        from sklearn.tree import plot_tree

        # Compute a figure size that scales with the number of nodes at max_depth
        width  = min(40, max(16, 2 ** max_depth * 3))
        height = max(8, max_depth * 3)

        fig, ax = plt.subplots(figsize=(width, height))

        plot_tree(
            model,
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,          # Colour nodes by majority class / predicted value
            rounded=True,         # Rounded node boxes for readability
            impurity=True,        # Show Gini / MSE at each node
            proportion=False,     # Show raw sample counts, not fractions
            fontsize=9,
            ax=ax,
        )

        ax.set_title(
            f"Decision Tree — {model_name} (max display depth = {max_depth})",
            fontsize=13,
            pad=12,
        )
        self._save(fig, "supervised", filename)

    def feature_importance_plot(self, importances: np.ndarray, feature_names: list, model_name: str, filename: str, top_n: int = 20) -> None:
        """
        Horizontal bar chart of the top-n most important features.

        top_n is automatically capped at the number of available features
        to prevent a shape mismatch when the dataset has fewer features
        than the requested top_n.
        """

        # Cap top_n to the actual number of available features
        top_n = min(top_n, len(feature_names))

        indices = np.argsort(importances)[::-1][:top_n]
        sorted_names  = [feature_names[i] for i in indices]
        sorted_scores = importances[indices]

        fig, ax = plt.subplots(figsize=(9, max(5, top_n // 2)))
        ax.barh(range(top_n), sorted_scores[::-1], color="steelblue")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(sorted_names[::-1], fontsize=9)
        ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13)
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        self._save(fig, "supervised", filename)

    def mkl_kernel_weights_plot(
        self,
        weights_natural: np.ndarray,
        weights_anti: np.ndarray,
        model_name: str,
        filename: str,
    ) -> None:
        """
        Plot the extremality-based kernel weight distributions for both the
        natural and anti-natural orderings side by side.
        Each bar represents one weak polynomial kernel.  In the natural ordering
        kernels with higher quality metrics (kernel alignment, FSM) receive larger
        weights; in the anti-natural ordering the weight is reversed and serves as
        a baseline for comparison.  Orange bars highlight the top-3 kernels in
        each panel.
        """
        n = len(weights_natural)
        indices = np.arange(n)
        fig, axes = plt.subplots(1, 2, figsize=(max(14, n), 5), sharey=False)

        for ax, weights, title_suffix in zip(
            axes,
            [weights_natural, weights_anti],
            ["Natural (high quality → high weight)",
             "Anti-Natural (low quality → high weight)"],
        ):
            bars = ax.bar(indices, weights, color="steelblue", edgecolor="white")

            # Highlight the top-3 kernels in a distinct colour
            top3 = np.argsort(weights)[::-1][:3]

            for idx in top3:
                bars[idx].set_color("darkorange")

            ax.set_title(
                f"{model_name} — {title_suffix}\n"
                "(orange = top-3 highest-weight kernels)",
                fontsize=11,
            )
            ax.set_xlabel("Kernel Index",      fontsize=10)
            ax.set_ylabel("Weight (sums to 1)", fontsize=10)
            ax.set_xticks(indices)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        self._save(fig, "supervised", filename)

    def mkl_kernel_metrics_plot(
        self,
        metrics_matrix: np.ndarray,
        metric_names: list,
        weights: np.ndarray,
        model_name: str,
        filename: str,
    ) -> None:
        """
        Plot per-kernel metric values as a grouped bar chart, with the natural
        weight overlaid as a line so the relationship between metric quality
        and assigned weight can be inspected visually.
        """
        n_kernels, n_metrics = metrics_matrix.shape
        x       = np.arange(n_kernels)
        width   = 0.8 / n_metrics          # bar width so groups don't overlap
        colours = plt.cm.tab10(np.linspace(0, 0.9, n_metrics))
        fig, ax1 = plt.subplots(figsize=(max(12, n_kernels), 5))

        for i, (name, colour) in enumerate(zip(metric_names, colours)):
            # Normalise metric values to [0, 1] for a fair visual comparison
            col = metrics_matrix[:, i]
            col_range = col.max() - col.min()
            col_norm  = (col - col.min()) / col_range if col_range > 0 else col
            ax1.bar(
                x + i * width - (n_metrics - 1) * width / 2,
                col_norm,
                width=width,
                label=name,
                color=colour,
                alpha=0.75,
                edgecolor="white",
            )

        ax1.set_xlabel("Kernel Index",              fontsize=10)
        ax1.set_ylabel("Normalised Metric Value",   fontsize=10)
        ax1.set_xticks(x)
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        # Overlay the natural weight on a secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(x, weights, color="black", marker="o", linewidth=1.8, markersize=5, label="Natural weight")
        ax2.set_ylabel("Natural Weight", fontsize=10)
        ax2.legend(loc="upper right", fontsize=9)
        ax1.set_title(
            f"Per-Kernel Metrics vs. Natural Weights — {model_name}",
            fontsize=12,
        )
        fig.tight_layout()
        self._save(fig, "supervised", filename)

    def model_comparison_bar(self, model_names: list, metric_values: list, metric_name: str, task: str, filename: str) -> None:
        """
        Horizontal bar chart comparing a single metric across models.
        """
        
        sorted_pairs = sorted(zip(metric_values, model_names))
        vals, names  = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(names, vals, color="steelblue", edgecolor="white")

        # Annotate each bar with its numeric value
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9,
            )

        ax.set_title(f"{task} Models — {metric_name} Comparison", fontsize=13)
        ax.set_xlabel(metric_name, fontsize=11)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.set_xlim(0, max(vals) * 1.15)
        self._save(fig, "supervised", filename)
