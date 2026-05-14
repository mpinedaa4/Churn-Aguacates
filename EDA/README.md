# EDA — Exploratory Data Analysis

## Purpose

Performs all feature engineering, data cleaning, descriptive statistics, and exploratory visualisations. It is the first processing step in the pipeline and produces the engineered DataFrame consumed by the Unsupervised module.

---

## File

`EDA/EDA.py`

---

## Class: `EDA`

### Constructor

```python
eda = EDA()
```

Instantiates `LoadData` and `Visualization` internally. The raw dataset is loaded immediately on construction.

### Method: `run() -> pd.DataFrame`

Executes the full EDA pipeline in order and returns the engineered, cleaned DataFrame.

```python
engineered_df = eda.run()
```

---

## Pipeline Steps

### Step 1 — Feature Engineering (`feature_engineering`)

Derives new variables from the raw columns and removes the originals.

| New Variable | Source | Description |
|---|---|---|
| `age` | `year_of_birth` | Computed as `2022 - year_of_birth`, clipped to [14, 80] |
| `pss_total` | `pss_1` to `pss_10` | Sum of all 10 PSS items (range 0 – 40) |
| `gad_total` | `gad_1` to `gad_7` | Sum of all 7 GAD items (range 0 – 21) |
| `phq9_total` | `phq_1` to `phq_9` | Sum of all 9 PHQ items (range 0 – 27) |
| `phq9_category` | `phq9_total` | Binned into five severity labels (see below) |
| `positive_affect` | 4 wellbeing items | Sum of happy, laughed, learned, enjoyment |
| `negative_affect` | 5 negative items | Sum of worried, depressed, angry, stress, lonely |

PHQ-9 category thresholds applied during binning:

| Category | Score Range |
|---|---|
| Minimal | 0 – 4 |
| Mild | 5 – 9 |
| Moderate | 10 – 14 |
| Moderately_Severe | 15 – 19 |
| Severe | 20 – 27 |

After engineering, the original item columns (`year_of_birth`, all `pss_*`, `gad_*`, `phq_*`, and the individual wellbeing items) are dropped.

### Step 2 — Data Cleaning (`clean_dataset`)

- Column names are stripped of leading/trailing whitespace and converted to lowercase.
- Missing values in `education_years_father` and `education_years_mother` are imputed with their respective column medians.
- A final integrity check confirms no remaining nulls; any remaining rows with nulls are dropped.

### Step 3 — Dataset Overview (`print_dataset_overview`)

Prints the total number of rows, columns, numeric variables, categorical variables, and a per-column null count to the console.

### Step 4 — Central Tendency (`central_tendency`)

Prints mean, median, and mode for all variables (mode is reported for every column; mean and median only for numeric columns).

### Step 5 — Dispersion Measures (`dispersion_measures`)

Prints range (max − min), variance, and standard deviation for all numeric columns.

### Step 6 — Position Measures (`position_measures`)

Prints Q1, Q2, Q3, IQR, and Tukey fence bounds (Q1 − 1.5×IQR, Q3 + 1.5×IQR) for all numeric columns. Outlier counts per column are reported.

### Step 7 — Histograms and Boxplots (`histograms`)

Saves a histogram (with mean and median reference lines, bin count from Sturges' rule) and a boxplot for each numeric variable.

Output directory: `graphs/eda/`

### Step 8 — Pie Charts (`pie_charts`)

Saves pie charts for the following low-cardinality categorical variables: `gender`, `socioeconomic_status`, `ethnicity`, `phq9_category`.

Output directory: `graphs/eda/`

### Step 9 — Correlation Analysis (`correlation_analysis`)

Prints Pearson correlation coefficients between every numeric feature and `phq9_total`. Saves a grid of scatter plots (feature vs. `phq9_total`) and a full lower-triangle Pearson correlation heatmap.

Output directory: `graphs/eda/`

### Step 10 — UMAP Projection (`umap_projection`)

Scales numeric features with `StandardScaler`, then fits a 3-D UMAP projection. The resulting embedding is plotted as a 3-D scatter plot coloured by `phq9_total`.

Output directory: `graphs/eda/`

---

## Key Constants

| Constant | Value | Description |
|---|---|---|
| `REFERENCE_YEAR` | 2022 | Used to compute participant age |
| `PHQ9_BINS` | [-1, 4, 9, 14, 19, 27] | Bin edges for PHQ-9 categories |
| `PHQ9_LABELS` | 5 severity strings | Labels for the above bins |
| `PIE_COLS` | 4 column names | Columns for which pie charts are generated |

---

## Output

Returns `self.clean_df` — the fully processed DataFrame — which is passed directly to `UnsupervisedLearning.run()` in `main.py`.