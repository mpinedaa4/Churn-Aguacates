# IndexScores — Metric Aggregation and Export

## Purpose

Aggregates all metrics produced by the Unsupervised and Supervised modules,
prints formatted leaderboards to the console, and exports the complete results
to a JSON file for external inspection or reproducibility.

---

## File

`IndexScores/indexes_scores.py`

---

## Class: `IndexesScoresEvaluator`

### Constructor

```python
evaluator = IndexesScoresEvaluator(metrics_dict=all_metrics, output_dir="output_metrics")
```

| Parameter | Default | Description |
|---|---|---|
| `metrics_dict` | — | Nested dictionary containing keys `Clustering`, `Regression`, and `Classification` |
| `output_dir` | `"output_metrics"` | Directory where the JSON export is written; created if it does not exist |

The `metrics_dict` is produced by merging the dictionary returned by
`UnsupervisedLearning.run()` with the one returned by
`PHQ9ModelTrainer.run_pipeline()`, as shown in `main.py`:

```python
all_metrics = {**supervised_metrics, **clustering_metrics}
```

---

## Method: `run_evaluation()`

Executes all display and export steps in order:

1. `display_clustering_scores()`
2. `display_regression_scores()`
3. `display_classification_scores()`
4. `display_sensitivity_specificity()`
5. `export_metrics_to_json()`

---

## Methods

### `display_clustering_scores()`

Prints a leaderboard for all five clustering models sorted by Silhouette Score
(descending). Columns displayed:

| Column | Description |
|---|---|
| `N_Clusters` | Number of clusters found by the model |
| `Silhouette` | Mean silhouette score; range [-1, +1]; higher is better |
| `Davies_Bouldin` | Davies-Bouldin index; range [0, ∞); lower is better |

Returns the summary as a `pd.DataFrame`, or `None` if no clustering data is present.

---

### `display_regression_scores()`

Prints a leaderboard for all regression models sorted by Validation R2
(descending). Validation and test metrics are shown side by side.

| Column | Description |
|---|---|
| `Val_R2` | Coefficient of Determination on the validation set |
| `Val_RMSE` | Root Mean Squared Error on the validation set |
| `Val_MAE` | Mean Absolute Error on the validation set |
| `Val_MSE` | Mean Squared Error on the validation set |
| `Test_R2` | Coefficient of Determination on the held-out test set |
| `Test_RMSE` | Root Mean Squared Error on the held-out test set |
| `Test_MAE` | Mean Absolute Error on the held-out test set |
| `Test_MSE` | Mean Squared Error on the held-out test set |

Returns the summary as a `pd.DataFrame`, or `None` if no regression data is present.

---

### `display_classification_scores()`

Prints a leaderboard for all classification models sorted by Validation
Accuracy (descending).

| Column | Description |
|---|---|
| `Val_Accuracy` | Fraction of correctly classified samples on the validation set |
| `Val_MacroF1` | Unweighted mean F1-score across all classes on the validation set |
| `Test_Accuracy` | Accuracy on the held-out test set |
| `Test_MacroF1` | Macro F1-score on the held-out test set |

Returns the summary as a `pd.DataFrame`, or `None` if no classification data is present.

---

### `display_sensitivity_specificity()`

Prints a per-class Sensitivity and Specificity table for every classification
model, for both the validation and test splits. These values are computed
in `supervised.py` using a one-vs-rest decomposition of the confusion matrix
and stored in the metrics dictionary under the key `Sensitivity_Specificity`.

Example output format:

```
  Model: Random_Forest_Clf
    [Validation]
    Class                    Sensitivity   Specificity
    -----------------------------------------------
    Minimal                       0.8200        0.9100
    Mild                          0.7500        0.8800
    ...
```

---

### `export_metrics_to_json()`

Serialises the full metrics dictionary to `output_dir/model_metrics.json`
with 4-space indentation. Numpy scalar types (`np.integer`, `np.floating`,
`np.ndarray`) are recursively converted to native Python types before
serialisation to prevent `TypeError` from `json.dump`.

---

## Metric Interpretation Reference

### Clustering

| Metric | Range | Direction | Interpretation |
|---|---|---|---|
| Silhouette Score | -1 to +1 | Higher is better | Measures how similar each sample is to its own cluster compared to other clusters. Values above 0.50 indicate strong structure. |
| Davies-Bouldin Index | 0 to ∞ | Lower is better | Ratio of within-cluster scatter to between-cluster separation. 0 indicates perfectly separated clusters. |

### Regression

| Metric | Range | Direction | Interpretation |
|---|---|---|---|
| R2 | (-∞, 1] | Higher is better | Proportion of variance in `phq9_total` explained by the model. 1 = perfect fit; 0 = no better than always predicting the mean; negative = worse than the mean. |
| RMSE | 0 to ∞ | Lower is better | Root Mean Squared Error. Expressed in the same units as `phq9_total` (PHQ-9 score points), making it directly interpretable. |
| MAE | 0 to ∞ | Lower is better | Mean Absolute Error. Average absolute deviation in PHQ-9 score points. Less sensitive to large individual errors than RMSE. |
| MSE | 0 to ∞ | Lower is better | Mean Squared Error. Penalises large errors more heavily than MAE due to squaring. |

### Classification

| Metric | Range | Direction | Interpretation |
|---|---|---|---|
| Accuracy | 0 to 1 | Higher is better | Fraction of all samples classified correctly. Can be misleading when classes are imbalanced; always examine alongside Macro F1. |
| Macro F1-Score | 0 to 1 | Higher is better | Unweighted average of per-class F1-scores. Gives equal weight to every class regardless of its size; the preferred summary metric for this imbalanced multiclass problem. |
| Sensitivity (Recall / TPR) | 0 to 1 | Higher is better | Per class: proportion of true members of that class correctly identified. A low sensitivity for "Severe" means many severe cases are missed. |
| Specificity (TNR) | 0 to 1 | Higher is better | Per class: proportion of true non-members correctly excluded. A low specificity for "Severe" means many non-severe cases are incorrectly flagged. |

---

## Output File

`output_metrics/model_metrics.json`

The file contains the full nested metrics dictionary. Example structure:

```json
{
    "Regression": {
        "Random_Forest_Reg": {
            "Validation": {"MSE": 4.21, "RMSE": 2.05, "MAE": 1.58, "R2": 0.74},
            "Test":       {"MSE": 4.87, "RMSE": 2.21, "MAE": 1.67, "R2": 0.71}
        }
    },
    "Classification": {
        "Random_Forest_Clf": {
            "Validation": {
                "Accuracy": 0.73,
                "Macro_F1": 0.68,
                "Sensitivity_Specificity": {
                    "Minimal": {"Sensitivity": 0.82, "Specificity": 0.91}
                }
            }
        }
    },
    "Clustering": {
        "KMeans": {"N_Clusters": 5, "Silhouette": 0.31, "Davies_Bouldin": 1.84}
    }
}
```