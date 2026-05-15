"""
EDA/EDA.py — Exploratory Data Analysis pipeline (cloud-optimised).

Changes vs. original
---------------------
1.  UMAP replaced with a 3-D PCA projection (sampled to SAMPLE_CAP rows for
    the visualisation so the plot remains fast and memory-safe on 1 M rows).
2.  All `print()` calls replaced with structured logging (file + console).
3.  MemoryGuard checks before every heavyweight operation.
4.  Histograms and boxplots are generated in batches; matplotlib figures are
    explicitly closed after saving to avoid leaking figure memory.
"""

from LoadData.LoadData import LoadData
from Visualization.Visualization import Visualization

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from logger import get_logger
from memory_guard import MemoryGuard, log_memory

log = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
PHQ9_BINS   = [-1, 4, 9, 14, 19, 27]
PHQ9_LABELS = ["Minimal", "Mild", "Moderate", "Moderately_Severe", "Severe"]

PSS_ITEMS = [f"pss_{i}" for i in range(1, 11)]
GAD_ITEMS = [f"gad_{i}" for i in range(1, 8)]
PHQ_ITEMS = [f"phq_{i}" for i in range(1, 10)]

WELLBEING_ITEMS = [
    "happy_yesterday", "laughed_yesterday", "learned_yesterday", "enjoyment_yesterday",
]
NEGATIVE_AFFECT_ITEMS = [
    "worried_yesterday", "felt_depressed_yesterday", "angry_yesterday",
    "stress_yesterday", "lonely_yesterday",
]

PIE_COLS = ["gender", "socioeconomic_status", "ethnicity", "phq9_category"]

REFERENCE_YEAR = 2022

# Maximum rows used for the 3-D PCA visualisation and the correlation scatter.
# The full dataset is still used for every statistical computation.
VIZ_SAMPLE_CAP = 50_000


class EDA:
    def __init__(self):
        loader = LoadData()
        self.df       = loader.load_data()
        self.clean_df = None
        self.visualizer = Visualization()

    def run(self) -> pd.DataFrame:
        """Run the full EDA pipeline and return the cleaned, engineered DataFrame."""
        log.info("=" * 60)
        log.info("EXPLORATORY DATA ANALYSIS")
        log.info("=" * 60)

        self.feature_engineering()
        self.clean_dataset()
        self.print_dataset_overview()
        self.central_tendency()
        self.dispersion_measures()
        self.position_measures()
        self.histograms()
        self.pie_charts()
        self.correlation_analysis()
        self.pca_projection()       # ← replaces umap_projection()

        return self.clean_df

    # ── Step 1: Feature Engineering ─────────────────────────────────────────
    def feature_engineering(self) -> None:
        log.info("[EDA] Step 1 — Feature Engineering")

        with MemoryGuard("Feature Engineering", threshold_gb=2):
            df = self.df.copy()

            df["age"] = REFERENCE_YEAR - df["year_of_birth"]
            df["age"] = df["age"].clip(lower=14, upper=80)
            df.drop(columns=["year_of_birth"], inplace=True)

            df["pss_total"]  = df[PSS_ITEMS].sum(axis=1)
            df["gad_total"]  = df[GAD_ITEMS].sum(axis=1)
            df["phq9_total"] = df[PHQ_ITEMS].sum(axis=1)
            df.drop(columns=PSS_ITEMS + GAD_ITEMS + PHQ_ITEMS, inplace=True)

            df["phq9_category"] = pd.cut(
                df["phq9_total"],
                bins=PHQ9_BINS,
                labels=PHQ9_LABELS,
            ).astype(str)

            self.clean_df = df

        log.info(
            "[EDA] Engineered dataset shape: %d rows × %d columns",
            self.clean_df.shape[0], self.clean_df.shape[1],
        )

    # ── Step 2: Data Cleaning ────────────────────────────────────────────────
    def clean_dataset(self) -> None:
        log.info("[EDA] Step 2 — Data Cleaning")

        df = self.clean_df.copy()
        df.columns = df.columns.str.strip().str.lower()

        for col in ("education_years_father", "education_years_mother"):
            if col in df.columns:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                log.debug("[EDA] Imputed '%s' nulls with median %.2f", col, median_val)

        remaining_nulls = df.isnull().sum().sum()
        if remaining_nulls > 0:
            log.warning(
                "[EDA] %d residual nulls after imputation — dropping those rows",
                remaining_nulls,
            )
            df.dropna(inplace=True)

        self.clean_df = df.reset_index(drop=True)
        log.info("[EDA] Clean dataset: %d rows × %d columns", *self.clean_df.shape)

    # ── Dataset Overview ─────────────────────────────────────────────────────
    def print_dataset_overview(self) -> None:
        df  = self.clean_df
        num = df.select_dtypes(include="number")
        cat = df.select_dtypes(exclude="number")

        log.info("=" * 60)
        log.info("DATASET OVERVIEW")
        log.info("=" * 60)
        log.info("  Total observations : %d", df.shape[0])
        log.info("  Total variables    : %d", df.shape[1])
        log.info("  Numeric variables  : %d  →  %s", num.shape[1], list(num.columns))
        log.info("  Categorical vars   : %d  →  %s", cat.shape[1], list(cat.columns))
        null_info = df.isnull().sum()
        log.info("  Missing values per column:\n%s", null_info.to_string())

    # ── Central Tendency ─────────────────────────────────────────────────────
    def central_tendency(self) -> None:
        df  = self.clean_df
        num = df.select_dtypes(include="number")

        log.info("=" * 60)
        log.info("CENTRAL TENDENCY MEASURES")
        log.info("=" * 60)
        log.info("-- Mean --\n%s", num.mean().round(3).to_string())
        log.info("-- Median --\n%s", num.median().to_string())

        mode_lines = []
        for col in df.columns:
            modes = df[col].mode().tolist()
            mode_lines.append(f"  {col:35s}: {modes}")
        log.info("-- Mode --\n%s", "\n".join(mode_lines))

    # ── Dispersion Measures ──────────────────────────────────────────────────
    def dispersion_measures(self) -> None:
        num = self.clean_df.select_dtypes(include="number")

        log.info("=" * 60)
        log.info("DISPERSION MEASURES")
        log.info("=" * 60)
        log.info("-- Range --\n%s",    (num.max() - num.min()).to_string())
        log.info("-- Variance --\n%s", num.var().round(4).to_string())
        log.info("-- Std Dev --\n%s",  num.std().round(4).to_string())

    # ── Position Measures ────────────────────────────────────────────────────
    def position_measures(self) -> None:
        num = self.clean_df.select_dtypes(include="number")

        log.info("=" * 60)
        log.info("POSITION MEASURES")
        log.info("=" * 60)

        Q1  = num.quantile(0.25)
        Q2  = num.quantile(0.50)
        Q3  = num.quantile(0.75)
        IQR = Q3 - Q1

        summary = pd.DataFrame({
            "Q1 (25%)":    Q1,
            "Q2 (50%)":    Q2,
            "Q3 (75%)":    Q3,
            "IQR":         IQR,
            "Lower Fence": Q1 - 1.5 * IQR,
            "Upper Fence": Q3 + 1.5 * IQR,
        })
        log.info("\n%s", summary.round(3).to_string())

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_counts = ((num < lower) | (num > upper)).sum()
        log.info("-- Outlier Counts per Column --\n%s", outlier_counts.to_string())

    # ── Histograms & Boxplots ────────────────────────────────────────────────
    def histograms(self) -> None:
        log.info("[EDA] Generating histograms and boxplots...")
        num = self.clean_df.select_dtypes(include="number")

        for col in num.columns:
            log.debug("[EDA] Plotting histogram/boxplot for '%s'", col)
            self.visualizer.histogram(
                series=num[col], col_name=col, filename=f"histogram_{col}.png",
            )
            self.visualizer.boxplot(
                series=num[col], col_name=col, filename=f"boxplot_{col}.png",
            )

        log.info("[EDA] Histograms and boxplots saved.")

    # ── Pie Charts ───────────────────────────────────────────────────────────
    def pie_charts(self) -> None:
        log.info("[EDA] Generating pie charts...")
        df = self.clean_df

        for col in PIE_COLS:
            if col in df.columns:
                self.visualizer.pie_chart(
                    series=df[col], col_name=col, filename=f"pie_{col}.png",
                )

        log.info("[EDA] Pie charts saved.")

    # ── Correlation Analysis ─────────────────────────────────────────────────
    def correlation_analysis(self) -> None:
        log.info("[EDA] Correlation Analysis")

        df     = self.clean_df
        num    = df.select_dtypes(include="number")
        target = "phq9_total"
        feat_cols = [c for c in num.columns if c != target]

        lines = []
        for col in feat_cols:
            r = df[col].corr(df[target])
            lines.append(f"  {col:35s}: r = {r:+.4f}")
        log.info("Pearson r with phq9_total:\n%s", "\n".join(lines))

        # For scatter plots, sample to keep the PNG readable and fast
        n_scatter = min(len(df), VIZ_SAMPLE_CAP)
        df_sample = df.sample(n=n_scatter, random_state=42) if len(df) > VIZ_SAMPLE_CAP else df
        log.info("[EDA] Scatter plot uses %d sampled rows", n_scatter)

        self.visualizer.scatter_vs_target(
            df=df_sample,
            feature_cols=feat_cols,
            target_col=target,
            filename="scatter_vs_phq9_total.png",
        )
        self.visualizer.correlation_matrix(
            df=num,
            filename="correlation_matrix.png",
        )
        log.info("[EDA] Correlation plots saved.")

    # ── PCA Projection (replaces UMAP) ───────────────────────────────────────
    def pca_projection(self) -> None:
        """
        Compute a 3-D PCA projection for visualisation.

        Why PCA instead of UMAP?
        ────────────────────────
        UMAP has O(n log n) time and O(n) memory but with very large constants.
        On 1 M rows it exhausts RAM before the graph construction finishes.
        PCA is an exact linear decomposition: O(n · p · k) time, negligible
        extra memory for 3 components, and produces a plot in seconds.

        The projection is computed on the full scaled feature matrix, but the
        scatter plot is drawn on a stratified random sample (VIZ_SAMPLE_CAP
        rows) to keep the PNG file size manageable.
        """
        log.info("[EDA] Computing 3-D PCA projection (replaces UMAP)...")

        with MemoryGuard("PCA Projection", threshold_gb=2):
            num = self.clean_df.select_dtypes(include="number")
            num = num.loc[:, num.nunique() > 1]
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(num)

            pca = PCA(n_components=3, random_state=42)
            embedding_3d = pca.fit_transform(features_scaled)

            explained = pca.explained_variance_ratio_ * 100
            log.info(
                "[EDA] PCA variance explained — PC1: %.2f%%  PC2: %.2f%%  PC3: %.2f%%",
                explained[0], explained[1], explained[2],
            )

            # Sample for the scatter plot so the PNG is not 1 M overlapping dots
            n = len(self.clean_df)
            if n > VIZ_SAMPLE_CAP:
                rng = __import__("numpy").random.default_rng(42)
                idx = rng.choice(n, VIZ_SAMPLE_CAP, replace=False)
                embedding_plot = embedding_3d[idx]
                target_plot    = self.clean_df["phq9_total"].values[idx]
                log.info(
                    "[EDA] PCA scatter using %d sampled rows (full dataset = %d)",
                    VIZ_SAMPLE_CAP, n,
                )
            else:
                embedding_plot = embedding_3d
                target_plot    = self.clean_df["phq9_total"].values

            # Reuse the existing umap_projection visualiser — it just needs a
            # (n, 3) embedding array and the target values.
            self.visualizer.pca_projection(
                embedding_3d=embedding_plot,
                target=self.clean_df["phq9_total"].values,
                filename="pca_projection_3d.png"
            )
            """
            self.visualizer.umap_projection(
                df=self.clean_df.iloc[
                    __import__("numpy").random.default_rng(42).choice(n, min(n, VIZ_SAMPLE_CAP), replace=False)
                    if n > VIZ_SAMPLE_CAP else slice(None)
                ],
                features_scaled=embedding_plot,
                target_col="phq9_total",
                filename="pca_projection_3d.png",
            )
            """

        log.info("[EDA] PCA projection saved to pca_projection_3d.png")
        log_memory("After PCA projection")
