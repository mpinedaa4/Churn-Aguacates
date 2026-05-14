# Mental Health Prediction Pipeline

A complete machine learning pipeline that predicts PHQ-9 depression severity categories from a student mental health survey dataset. The pipeline covers exploratory data analysis, unsupervised clustering, supervised learning, and metric reporting. The dataset is retrieved from a 2022 research developed by the ICESI University.

---

## Project Structure

```
project_root/
|
|-- main.py                    # Entry point; orchestrates all pipeline steps
|
|-- Dataset/
|   |-- dataset.csv            # Raw survey data
|   `-- README.md
|
|-- EDA/
|   |-- EDA.py                 # Feature engineering and exploratory analysis
|   `-- README.md
|
|-- LoadData/
|   |-- LoadData.py            # CSV loading utility
|   `-- README.md
|
|-- Visualization/
|   |-- Visualization.py       # All plotting helpers for every pipeline stage
|   `-- README.md
|
|-- Unsupervised/
|   |-- Unsupervised.py        # Clustering models and dataset relabelling
|   `-- README.md
|
|-- Supervised/
|   |-- supervised.py          # Regression and classification models
|   `-- README.md
|
|-- IndexScores/
|   |-- indexes_scores.py      # Metric aggregation and JSON export
|   `-- README.md
|
|-- graphs/                    # All generated figures (created at runtime)
|   |-- eda/
|   |-- unsupervised/
|   `-- supervised/
|
`-- output_metrics/
    `-- model_metrics.json     # Full metrics export (created at runtime)
```

---

## Pipeline Overview

The pipeline executes four sequential steps, each building on the output of the previous one.

**Step 1 — EDA (`EDA/EDA.py`)**
Loads the raw dataset, derives composite questionnaire scores, computes participant age, creates the PHQ-9 categorical label, handles missing values, and runs all descriptive statistics and visualisations.

**Step 2 — Unsupervised Learning (`Unsupervised/Unsupervised.py`)**
Applies five clustering algorithms (K-Means, Fuzzy C-Means, DBSCAN, Subtractive, Agglomerative) to the engineered dataset without using the label. The optimal number of clusters is determined via the K-Means elbow method. Cluster IDs are then mapped to PHQ-9 severity categories based on median `phq9_total` scores, producing a relabelled dataset for supervised training.

**Step 3 — Supervised Learning (`Supervised/supervised.py`)**
Trains four regression models (predicting `phq9_total` as a continuous variable) and seven classification models (predicting `cluster_label` as a categorical variable) on the relabelled dataset. Data is split 60/20/20 into training, validation, and test sets.

**Step 4 — Metric Export (`IndexScores/indexes_scores.py`)**
Aggregates all clustering, regression, and classification metrics into a single dictionary, prints formatted leaderboards to the console, and exports everything to `output_metrics/model_metrics.json`.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Execute from the project root directory:

```bash
python main.py
```

All figures are saved under `graphs/` and the JSON metrics file under `output_metrics/`.

---

## Dependencies

| Package | Purpose |
|---|---|
| numpy | Numerical computation |
| pandas | DataFrame manipulation |
| scikit-learn | Machine learning models and preprocessing |
| matplotlib | Figure rendering |
| seaborn | Statistical visualisations |
| umap-learn | UMAP dimensionality reduction |
| scikit-fuzzy | Fuzzy C-Means clustering |

Additionaly, you should install the custom library we made for multiple kernel learning (MKL) using the following command:
```bash
pip install -e MKL/
```

---

## Output Files

| Path | Contents |
|---|---|
| `graphs/eda/` | Histograms, boxplots, pie charts, scatter plots, correlation matrix, UMAP projection |
| `graphs/unsupervised/` | Elbow curve, silhouette plot, cluster scatter plots for all five algorithms |
| `graphs/supervised/` | Confusion matrices, predicted-vs-actual scatter plots, feature importance charts, decision tree diagrams, model comparison bar charts |
| `output_metrics/model_metrics.json` | All numeric metrics for every model, both validation and test sets |