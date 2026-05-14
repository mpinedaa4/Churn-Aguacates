# Supervised Learning

## Purpose

Trains and evaluates regression and classification models on the relabelled
dataset produced by the Unsupervised module. Regression models predict the
continuous PHQ-9 total score; classification models predict the PHQ-9
severity category assigned during clustering.

---

## File

`Supervised/supervised.py`

---

## Class: `PHQ9ModelTrainer`

### Constructor

```python
trainer = PHQ9ModelTrainer(
    target_reg="phq9_total",
    target_clf="cluster_label",
    random_state=42,
)
```

| Parameter | Default | Description |
|---|---|---|
| `target_reg` | `"phq9_total"` | Continuous regression target column |
| `target_clf` | `"cluster_label"` | Categorical classification target column |
| `random_state` | `42` | Seed for all stochastic operations |

### Method: `run_pipeline(df) -> dict`

Executes the full supervised learning pipeline and returns a metrics dictionary.

```python
supervised_metrics = trainer.run_pipeline(relabeled_df)
```

**Input:** The relabelled DataFrame from `UnsupervisedLearning.run()`.

**Returns:** A nested dictionary with keys `"Regression"` and
`"Classification"`, each containing per-model metrics for both validation
and test splits.

---

## Data Split Strategy

The dataset is partitioned into three disjoint sets using a two-stage split:

```
Full dataset
    └── 80% temporary  ──┬── 60% training   (fit model parameters)
                         └── 20% validation  (compare models, tune)
    └── 20% test                             (final unbiased evaluation)
```

Stratification on `cluster_label` is applied at both splits to ensure every
PHQ-9 severity class is proportionally represented in all three partitions.

Feature scaling is performed with `StandardScaler` fitted exclusively on the
training set. The same fitted scaler is then applied to the validation and
test sets to prevent data leakage.

---

## Models

### Regression Models

All regression models predict `phq9_total` as a continuous numeric value.

| Model | Key Notes |
|---|---|
| Linear Regression | Fits a weighted sum of features; assumes a linear relationship with the target. Serves as a simple interpretable baseline. |
| Decision Tree Regressor | Partitions the feature space into rectangular regions by recursively splitting on the most informative feature. No assumptions about linearity; prone to overfitting without depth constraints. A diagram of the trained tree is saved automatically. |
| Random Forest Regressor | Ensemble of 100 decision trees trained on bootstrap samples with random feature subsets. Reduces the variance of individual trees through averaging. |
| Gradient Boosting Regressor | Additive ensemble that trains each tree to correct the residual errors of the previous one. Generally achieves high accuracy but requires more tuning than Random Forest. |

### Classification Models

All classification models predict `cluster_label` (the PHQ-9 severity category
string). The `class_weight="balanced"` argument is used where available to
compensate for class imbalance.

| Model | Key Notes |
|---|---|
| Logistic Regression | A linear classifier using the softmax function for multiclass output. `max_iter=2000` is set to ensure convergence on this dataset. |
| Decision Tree Classifier | Same tree-partitioning logic as the regressor, but predicts the majority class in each leaf. A diagram of the trained tree is saved automatically. |
| Random Forest Classifier | Ensemble of 100 balanced decision trees. More robust to overfitting than a single tree. |
| Gradient Boosting Classifier | Sequential boosting ensemble; each stage fits the pseudo-residuals of the current ensemble. |
| SVM Linear | Support Vector Machine with a linear kernel. Finds the hyperplane that maximises the margin between classes. |
| SVM Polynomial | SVM with a degree-3 polynomial kernel. Captures non-linear interactions between features. |
| SVM RBF | SVM with a Radial Basis Function kernel. Effective in high-dimensional feature spaces; `gamma="scale"` sets gamma to `1 / (n_features * X.var())`. |

---

## Decision Tree Visualisation

After training, both the regression and classification Decision Tree models are
passed to `Visualization.decision_tree_plot()`, which uses
`sklearn.tree.plot_tree` to render the branching structure of the tree.

Each internal node in the diagram shows:

- **Feature name and threshold** — the variable and cut-point used to split
  samples at that node (e.g., `pss_total <= 14.5`).
- **Impurity** — Gini impurity (classifiers) or Mean Squared Error
  (regressors) at that node, measuring how mixed the samples are before
  the split. A value of 0 means all samples belong to one class.
- **Samples** — number of training samples reaching that node.
- **Value** — for classifiers, the count of samples per class; for
  regressors, the mean target value.
- **Class** (classifiers only) — the majority class at that node, which is
  the prediction if a sample stops there.

Because fully grown trees can have hundreds of nodes, the diagrams are
rendered to a configurable maximum depth (default: `max_depth=4`), showing
only the most important early splits. The features appearing highest in the
tree are those the model considers most informative for predicting PHQ-9
severity.

Output files:

| File | Description |
|---|---|
| `graphs/supervised/tree_plot_Decision_Tree_Reg.png` | Regression tree diagram |
| `graphs/supervised/tree_plot_Decision_Tree_Clf.png` | Classification tree diagram |

---

## Metrics Computed

Metrics are computed on both the **validation set** (used for model comparison)
and the **test set** (final unbiased evaluation). All values are stored in the
returned dictionary and later displayed by `IndexesScoresEvaluator`.

### Regression Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| MSE | mean((y - y_hat)^2) | Average squared error. Penalises large errors heavily. Lower is better. |
| RMSE | sqrt(MSE) | Same units as `phq9_total` (PHQ-9 points). Easier to interpret than MSE. Lower is better. |
| MAE | mean(|y - y_hat|) | Average absolute deviation in PHQ-9 points. Less sensitive to outliers than RMSE. Lower is better. |
| R2 | 1 - SS_res / SS_tot | Proportion of variance in the target explained by the model. Range (-∞, 1]; higher is better. |

### Classification Metrics

| Metric | Interpretation |
|---|---|
| Accuracy | Fraction of all samples classified correctly. Can be misleading with imbalanced classes; use alongside Macro F1. |
| Macro F1-Score | Unweighted average of per-class F1-scores. Preferred summary metric for this problem because it gives equal weight to every severity class regardless of how many samples it contains. |
| Sensitivity (per class) | True Positive Rate: of all samples truly in class C, what fraction was correctly identified? A low sensitivity for "Severe" means many genuinely severe cases go undetected. |
| Specificity (per class) | True Negative Rate: of all samples truly not in class C, what fraction was correctly excluded? A low specificity means many non-severe cases are incorrectly flagged as belonging to class C. |

Sensitivity and specificity are derived from a one-vs-rest decomposition of
the confusion matrix: for each class C, TP, FN, FP, and TN are computed by
treating C as the positive class and all others as negative.

---

## Generated Plots

All figures are saved to `graphs/supervised/`.

| File pattern | Description |
|---|---|
| `reg_scatter_<model>.png` | Predicted vs. actual scatter for each regression model |
| `feature_importance_<model>.png` | Top feature importances for tree-based models |
| `tree_plot_<model>.png` | Decision tree branch diagram (Decision Tree models only) |
| `cm_val_<model>.png` | Confusion matrix on the validation set |
| `cm_test_<model>.png` | Confusion matrix on the test set |
| `reg_model_comparison_r2.png` | Bar chart comparing all regression models by R2 |
| `clf_model_comparison_accuracy.png` | Bar chart comparing all classification models by accuracy |