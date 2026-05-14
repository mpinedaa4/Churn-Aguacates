# Unsupervised Learning

## Purpose

Applies five clustering algorithms to the engineered dataset (without using the PHQ-9 label) to discover the natural grouping structure. The optimal number of clusters is determined from the K-Means elbow method. Cluster IDs are then mapped to PHQ-9 severity category names, producing a relabelled dataset for supervised training.

---

## File

`Unsupervised/Unsupervised.py`

---

## What the Silhouette Score Measures

The silhouette score quantifies how well each sample fits within its assigned cluster compared to neighbouring clusters. It is defined for each sample as:

```
s = (b - a) / max(a, b)
```

where `a` is the mean distance from the sample to all other points in its own cluster, and `b` is the mean distance from the sample to all points in the nearest other cluster.

The mean silhouette score across all samples ranges from **-1 to +1**:

| Range | Interpretation |
|---|---|
| 0.71 – 1.00 | Strong cluster structure |
| 0.51 – 0.70 | Reasonable cluster structure |
| 0.26 – 0.50 | Weak cluster structure; clusters may overlap |
| Below 0.25 | No meaningful cluster structure |
| Negative | Samples may have been assigned to the wrong cluster |

A high silhouette score means samples are compact within their own cluster and well-separated from other clusters. It is used alongside the elbow plot to confirm the choice of k.

---

## Class: `UnsupervisedLearning`

### Constructor

```python
unsupervised = UnsupervisedLearning(random_state=42)
```

| Parameter | Default | Description |
|---|---|---|
| `random_state` | `42` | Seed for all stochastic operations |

### Method: `run(df) -> tuple`

Executes the full clustering pipeline and returns a tuple of `(relabeled_df, metrics_dict)`.

```python
relabeled_df, clustering_metrics = unsupervised.run(engineered_df)
```

**Input:** The engineered DataFrame from `EDA.run()`. Columns `phq9_category` and `phq9_total` are excluded from the feature matrix used for clustering.

**Returns:**
- `relabeled_df`: The input DataFrame with an additional `cluster_label` column containing PHQ-9 severity strings.
- `clustering_metrics`: A dictionary structured as `{"Clustering": {model_name: {metrics}}}`.

---

## Models

### K-Means

Partitions the dataset into k non-overlapping clusters by minimising within-cluster sum of squares (inertia). k is searched over the range [2, 10].

**Elbow method:** Inertia is recorded for each k. The optimal k is detected as the point of maximum curvature in the inertia curve, computed as the index of the maximum value in the second derivative of inertia.

**Plots generated:**
- `graphs/unsupervised/elbow_plot.png` — Inertia vs. k
- `graphs/unsupervised/silhouette_scores.png` — Silhouette score vs. k
- `graphs/unsupervised/kmeans_clusters.png` — 2-D PCA cluster scatter

### Fuzzy C-Means

A soft generalisation of K-Means in which each sample holds a membership degree to every cluster rather than a hard assignment. The fuzziness exponent `m = 2.0` controls the degree of overlap. The cluster with the maximum membership degree is taken as the hard label for evaluation purposes.

Uses the same `c` (number of clusters) as the optimal k from K-Means.

**Plot generated:** `graphs/unsupervised/fuzzy_cmeans_clusters.png`

### DBSCAN

A density-based algorithm that identifies clusters as dense regions separated by lower-density areas. It does not require specifying the number of clusters in advance and can label low-density points as noise (label = -1).

**Epsilon estimation:** The 4th-nearest-neighbour distances are computed for all points. The 1st percentile of these distances (sorted descending) is used as epsilon, with a lower bound of 0.5 to prevent degenerate results.

**Metrics** are computed only on non-noise points. If fewer than two clusters are found, silhouette and Davies-Bouldin scores are set to `None`.

**Plot generated:** `graphs/unsupervised/dbscan_clusters.png`

### Subtractive Clustering

An algorithm that estimates cluster centres by computing a potential function over the data. The point with the highest potential becomes the first centre; the neighbourhood around it is suppressed, and the process repeats until the remaining potential falls below a threshold.

Parameters: `r_a = 0.5` (neighbourhood radius), `r_b = 0.75` (suppression radius), `eps_upper = 0.5`, `eps_lower = 0.15`.

For performance, the algorithm is run on a random subsample of up to 2,000 points. The identified centres are then used to assign all samples in the full dataset to the nearest centre.

**Plot generated:** `graphs/unsupervised/subtractive_clusters.png`

### Agglomerative Clustering

A hierarchical bottom-up algorithm. Each sample starts in its own cluster; clusters are iteratively merged using Ward linkage, which minimises the total within-cluster variance at each merge step. The number of clusters is set to the optimal k from K-Means.

**Plot generated:** `graphs/unsupervised/agglomerative_clusters.png`

---

## Evaluation Metrics

All models (except DBSCAN in degenerate cases) are evaluated with:

| Metric | Interpretation |
|---|---|
| Silhouette Score | Range [-1, +1]; higher is better. Measures compactness and separation. |
| Davies-Bouldin Index | Range [0, ∞); lower is better. Ratio of within-cluster scatter to between-cluster separation. |

---

## Relabelling Procedure

After all models have run, the relabelling step maps integer cluster IDs from the final K-Means model to PHQ-9 severity category names:

1. K-Means is re-fitted with the optimal k on the full scaled dataset.
2. The median `phq9_total` is computed for each cluster.
3. Each cluster is assigned the PHQ-9 severity category whose score range contains that median.
4. The resulting string labels are written to a new column `cluster_label` in the output DataFrame.

This procedure ensures that the cluster labels are clinically interpretable and directly comparable to the original PHQ-9 categories, even when the number of clusters does not match the number of categories.

---

## Key Constants

| Constant | Value | Description |
|---|---|---|
| `PHQ9_CATEGORIES` | 5 severity strings | Ordered category names for relabelling |
| `N_PHQ9_CATS` | 5 | Default k if elbow detection fails |
| `K_MIN` | 2 | Minimum k in elbow search |
| `K_MAX` | 10 | Maximum k in elbow search |