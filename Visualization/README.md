# Visualization

## Purpose

Centralises all figure generation for the entire pipeline. Every other module calls methods on a `Visualization` instance rather than constructing plots directly, keeping plotting logic in one place and ensuring a consistent visual style.

---

## File

`Visualization/Visualization.py`

---

## Output Directory Structure

All figures are saved under a configurable root directory (default: `graphs/`), organised into three subdirectories that are created automatically at construction time.

```
graphs/
|-- eda/
|-- unsupervised/
`-- supervised/
```

---

## Class: `Visualization`

### Constructor

```python
vis = Visualization(output_dir="graphs")
```

| Parameter | Default | Description |
|---|---|---|
| `output_dir` | `"graphs"` | Root directory for all saved figures |

---

## EDA Methods

### `histogram(series, col_name, filename)`

Plots a histogram with mean (red dashed) and median (green solid) vertical reference lines. The number of bins is determined by Sturges' rule: `k = 1 + log2(n)`, with a minimum of 5.

Saved to: `graphs/eda/<filename>`

### `boxplot(series, col_name, filename)`

Plots a single vertical boxplot for outlier inspection.

Saved to: `graphs/eda/<filename>`

### `pie_chart(series, col_name, filename)`

Plots a pie chart with percentage labels for a low-cardinality categorical variable.

Saved to: `graphs/eda/<filename>`

### `correlation_matrix(df, filename)`

Plots a lower-triangle Pearson correlation heatmap for all numeric columns in `df`. Annotations are omitted for readability when many features are present.

Saved to: `graphs/eda/<filename>`

### `scatter_vs_target(df, feature_cols, target_col, filename)`

Plots a grid of scatter plots (one per feature) against the target variable. Each subplot is annotated with the Pearson correlation coefficient.

Saved to: `graphs/eda/<filename>`

### `umap_projection(df, features_scaled, target_col, filename)`

Computes a 3-D UMAP embedding from a pre-scaled feature matrix and plots the result as a 3-D scatter plot coloured by the target column.

Saved to: `graphs/eda/<filename>`

---

## Unsupervised Methods

### `elbow_plot(k_values, inertias, filename)`

Plots the K-Means elbow curve: inertia (within-cluster sum of squares) against the number of clusters. The elbow point indicates the optimal k.

Saved to: `graphs/unsupervised/<filename>`

### `silhouette_plot(k_values, silhouettes, filename)`

Plots mean silhouette score against number of clusters, complementing the elbow curve with a quality-of-separation measure.

Saved to: `graphs/unsupervised/<filename>`

### `cluster_scatter_2d(embedding_2d, labels, title, filename)`

Plots a 2-D PCA scatter plot coloured by cluster labels. Noise points (label = -1, produced by DBSCAN) are shown in a distinct colour and labelled "Noise".

Saved to: `graphs/unsupervised/<filename>`

---

## Supervised Methods

### `confusion_matrix_plot(cm_array, class_names, model_name, filename)`

Plots a labelled confusion matrix heatmap. True labels are on the y-axis; predicted labels on the x-axis.

Saved to: `graphs/supervised/<filename>`

### `regression_scatter(y_true, y_pred, model_name, filename)`

Plots predicted values against actual values for a regression model. A dashed red diagonal represents the perfect-prediction reference line. Deviations from this line indicate prediction error.

Saved to: `graphs/supervised/<filename>`

### `feature_importance_plot(importances, feature_names, model_name, filename, top_n)`

Plots a horizontal bar chart of the top `top_n` most important features for tree-based models. `top_n` is automatically capped at the number of available features to prevent shape errors.

Saved to: `graphs/supervised/<filename>`

### `decision_tree_plot(model, feature_names, class_names, model_name, filename, max_depth)`

Plots the structure of a trained `DecisionTreeClassifier` or `DecisionTreeRegressor` using `sklearn.tree.plot_tree`. Only the first `max_depth` levels are shown to keep the diagram readable. Each node displays the splitting feature, threshold, impurity, sample count, and (for classifiers) the majority class.

Saved to: `graphs/supervised/<filename>`

### `model_comparison_bar(model_names, metric_values, metric_name, task, filename)`

Plots a horizontal bar chart comparing one metric across all models in a task (regression or classification). Bars are sorted ascending so the best-performing model appears at the top. Each bar is annotated with its numeric value.

Saved to: `graphs/supervised/<filename>`

---

## Shared Style Constants

| Constant | Value | Description |
|---|---|---|
| `PALETTE_MAIN` | `"viridis"` | Colormap for continuous-target scatter plots |
| `DPI` | `150` | Resolution for all saved figures |