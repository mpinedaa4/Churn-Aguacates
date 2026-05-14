from LoadData.LoadData import LoadData
from Visualization.Visualization import Visualization
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Constants
PHQ9_BINS = [-1, 4, 9, 14, 19, 27] # Thresholds for the PHQ-9 scale
PHQ9_LABELS = ["Minimal", "Mild", "Moderate", "Moderately_Severe", "Severe"] # Interpretation according to PHQ-9 scale

# Columns whose items are aggregated into composite scores
PSS_ITEMS = [f"pss_{i}" for i in range(1, 11)] # 10-item Perceived Stress Scale
GAD_ITEMS = [f"gad_{i}" for i in range(1, 8)] # 7-item Generalised Anxiety Disorder scale
PHQ_ITEMS = [f"phq_{i}" for i in range(1, 10)] # 9-item Patient Health Questionnaire (depression)

# Wellbeing items collected "yesterday"
WELLBEING_ITEMS = [
    "happy_yesterday", "laughed_yesterday", "learned_yesterday", "enjoyment_yesterday",
]
NEGATIVE_AFFECT_ITEMS = [
    "worried_yesterday", "felt_depressed_yesterday", "angry_yesterday",
    "stress_yesterday", "lonely_yesterday",
]

# Variables suitable for pie charts (low cardinality after engineering)
PIE_COLS = ["gender", "socioeconomic_status", "ethnicity", "phq9_category"]

# Reference year for age computation
REFERENCE_YEAR = 2022

class EDA:
    def __init__(self):
        loader = LoadData()
        self.df = loader.load_data()
        self.clean_df = None
        self.visualizer = Visualization()

    def run(self) -> pd.DataFrame:
        """
        Runs the full EDA pipeline in the correct order.
        """
        print("EXPLORATORY DATA ANALYSIS")

        self.feature_engineering()
        self.clean_dataset()
        self.print_dataset_overview()
        self.central_tendency()
        self.dispersion_measures()
        self.position_measures()
        self.histograms()
        self.pie_charts()
        self.correlation_analysis()
        #self.umap_projection()

        return self.clean_df
    def feature_engineering(self) -> None:
        """
        Calculate test scores, calculate age of participants, and create the dataset label
        """
        print("\n[EDA] Step 1 — Feature Engineering")
        df = self.df.copy()

        # -- Age (as of reference year) --
        df["age"] = REFERENCE_YEAR - df["year_of_birth"]

        # data-entry errors; cap age at a upper bound of 80 and lower bound of 14.
        df["age"] = df["age"].clip(lower=14, upper=80)
        df.drop(columns=["year_of_birth"], inplace=True)

        # Calculate total score for tests
        df["pss_total"] = df[PSS_ITEMS].sum(axis=1)
        df["gad_total"] = df[GAD_ITEMS].sum(axis=1)
        df["phq9_total"] = df[PHQ_ITEMS].sum(axis=1)
        df.drop(columns=PSS_ITEMS + GAD_ITEMS + PHQ_ITEMS, inplace=True)

        # PHQ-9 categorical label (target)
        df["phq9_category"] = pd.cut(
            df["phq9_total"],
            bins=PHQ9_BINS,
            labels=PHQ9_LABELS,
        ).astype(str)

        self.clean_df = df
        print(
            f"[EDA] Engineered dataset shape: "
            f"{self.clean_df.shape[0]} rows x {self.clean_df.shape[1]} columns"
        )

    def clean_dataset(self) -> None:
        """
        Standardise and clean the engineered dataset.
        """
        print("\n[EDA] Step 2 — Data Cleaning")
        df = self.clean_df.copy()

        # Column name normalisation
        df.columns = df.columns.str.strip().str.lower()

        # Impute parental education years with the column median
        for col in ("education_years_father", "education_years_mother"):
            if col in df.columns:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

        # Integrity check
        remaining_nulls = df.isnull().sum().sum()
        if remaining_nulls > 0:
            df.dropna(inplace=True)

        self.clean_df = df.reset_index(drop=True)

    def print_dataset_overview(self) -> None:
        """Print basic dataset size and variable type information."""
        df = self.clean_df
        num = df.select_dtypes(include="number")
        cat = df.select_dtypes(exclude="number")

        print("\n" + "=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)
        print(f"  Total observations : {df.shape[0]}")
        print(f"  Total variables    : {df.shape[1]}")
        print(f"  Numeric variables  : {num.shape[1]}  →  {list(num.columns)}")
        print(f"  Categorical vars   : {cat.shape[1]}  →  {list(cat.columns)}")
        print(f"\n  Missing values per column:\n{df.isnull().sum().to_string()}")

    def central_tendency(self) -> None:
        """Compute and print mean, median, and mode for each variable."""
        df  = self.clean_df
        num = df.select_dtypes(include="number")

        print("\n" + "=" * 60)
        print("CENTRAL TENDENCY MEASURES")
        print("=" * 60)

        # Mean
        print("\n  -- Mean (numeric variables) --")
        print(num.mean().round(3).to_string())

        # Median
        print("\n  -- Median (numeric variables) --")
        print(num.median().to_string())

        # Mode
        print("\n  -- Mode --")
        for col in df.columns:
            modes = df[col].mode().tolist()
            print(f"  {col:35s}: {modes}")

    def dispersion_measures(self) -> None:
        """Compute and print range, variance, and standard deviation."""
        num = self.clean_df.select_dtypes(include="number")

        print("\n" + "=" * 60)
        print("DISPERSION MEASURES")
        print("=" * 60)

        # Range
        print("\n  -- Range --")
        print((num.max() - num.min()).to_string())

        # Variance
        print("\n  -- Variance --")
        print(num.var().round(4).to_string())

        # Standard deviation
        print("\n  -- Standard Deviation --")
        print(num.std().round(4).to_string())

    def position_measures(self) -> None:
        """
        Compute quartiles, IQR and identify potential outliers
        (Tukey fence criterion: Q1 - 1.5*IQR, Q3 + 1.5*IQR).
        """
        num = self.clean_df.select_dtypes(include="number")

        print("\n" + "=" * 60)
        print("POSITION MEASURES")
        print("=" * 60)

        Q1 = num.quantile(0.25)
        Q2 = num.quantile(0.50)
        Q3 = num.quantile(0.75)
        IQR = Q3 - Q1

        summary = pd.DataFrame({
            "Q1 (25%)": Q1,
            "Q2 (50%)": Q2,
            "Q3 (75%)": Q3,
            "IQR":      IQR,
            "Lower Fence": Q1 - 1.5 * IQR,
            "Upper Fence": Q3 + 1.5 * IQR,
        })
        print(f"\n{summary.round(3).to_string()}")

        # Outlier counts per column
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_counts = ((num < lower) | (num > upper)).sum()

        print("\n  -- Outlier Counts per Column --")
        print(outlier_counts.to_string())

    def histograms(self) -> None:
        """Save a histogram and boxplot for each numeric variable."""

        print("\n[EDA] Generating histograms and boxplots...")
        num = self.clean_df.select_dtypes(include="number")

        for col in num.columns:
            self.visualizer.histogram(
                series=num[col],
                col_name=col,
                filename=f"histogram_{col}.png",
            )
            self.visualizer.boxplot(
                series=num[col],
                col_name=col,
                filename=f"boxplot_{col}.png",
            )

    def pie_charts(self) -> None:
        """Save pie charts for low-cardinality categorical columns."""

        print("[EDA] Generating pie charts...")
        df = self.clean_df

        for col in PIE_COLS:
            if col in df.columns:
                self.visualizer.pie_chart(
                    series=df[col],
                    col_name=col,
                    filename=f"pie_{col}.png",
                )

    def correlation_analysis(self) -> None:
        """
        Print Pearson correlation coefficients between each numeric feature
        and phq9_total, and save a full correlation matrix heat-map.
        """

        print("\n[EDA] Correlation Analysis")

        df = self.clean_df
        num = df.select_dtypes(include="number")
        target = "phq9_total"
        feat_cols = [c for c in num.columns if c != target]

        # Pearson correlation coefficients
        print("\nPearson r with phq9_total")
        for col in feat_cols:
            r = df[col].corr(df[target])
            print(f"  {col:35s}: r = {r:+.4f}")

        # Scatter plots vs. target
        self.visualizer.scatter_vs_target(
            df=df,
            feature_cols=feat_cols,
            target_col=target,
            filename="scatter_vs_phq9_total.png",
        )

        # Full correlation heat-map
        self.visualizer.correlation_matrix(
            df=num,
            filename="correlation_matrix.png",
        )

    def umap_projection(self) -> None:
        """Scale numeric features and render a 3-D UMAP projection."""

        print("[EDA] Computing UMAP projection...")
        num = self.clean_df.select_dtypes(include="number")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(num)

        self.visualizer.umap_projection(
            df=self.clean_df,
            features_scaled=features_scaled.astype(np.float32),
            target_col="phq9_total",
            filename="umap.png",
        )