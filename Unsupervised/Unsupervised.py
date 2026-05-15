"""
Unsupervised/Unsupervised.py — Clustering pipeline (cloud-optimised).

Changes vs. original
---------------------
1.  Agglomerative Clustering — replaced with mini-batch / approximate approach:
      • On datasets > AGG_SAMPLE_THRESHOLD rows, Ward linkage is run on a
        stratified sample (AGG_MAX_SAMPLES rows), then every remaining point
        is assigned to its nearest centroid via a simple distance lookup.
        This cuts memory from O(n²) (the full linkage dendrogram) to O(s²)
        where s = AGG_MAX_SAMPLES ≪ n.
      • On small datasets (≤ AGG_SAMPLE_THRESHOLD) the original full
        AgglomerativeClustering is used unchanged.

2.  Subtractive Clustering — already sampled (MAX_SAMPLES = 2 000) in the
    original; the cap is kept and memory is guarded.

3.  DBSCAN — epsilon estimation now subsamples the k-NN graph so the
    NearestNeighbors fit does not try to hold 1 M distance rows in RAM.

4.  Fuzzy C-Means — `skfuzzy` holds the full (n × c) membership matrix.
    For datasets > FCM_SAMPLE_THRESHOLD a stratified sample is used for
    fitting; every remaining point is assigned to the nearest centroid.

5.  K-Means — unchanged; sklearn's KMeans is already mini-batch friendly
    (n_init=10); we guard memory before the silhouette computation because
    sample_size is already capped at 2 000.

6.  All `print()` → structured logging.  Memory is logged before and after
    every fit step.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import skfuzzy as fuzz
    _HAS_SKFUZZY = True
except ImportError:
    _HAS_SKFUZZY = False

from Visualization.Visualization import Visualization
from logger import get_logger
from memory_guard import MemoryGuard, log_memory, check_memory

log = get_logger(__name__)

# ── Category definitions ─────────────────────────────────────────────────────
PHQ9_CATEGORIES = ["Minimal", "Mild", "Moderate", "Moderately_Severe", "Severe"]
N_PHQ9_CATS     = len(PHQ9_CATEGORIES)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
K_MIN, K_MAX = 2, 10

# Agglomerative: above this threshold use sampled Ward
AGG_SAMPLE_THRESHOLD = 50_000
AGG_MAX_SAMPLES      = 40_000

# Fuzzy C-Means: above this threshold use sampled fitting
FCM_SAMPLE_THRESHOLD = 100_000
FCM_MAX_SAMPLES      = 50_000

# DBSCAN k-NN graph estimation: cap to avoid memory spike
KNN_DBSCAN_CAP = 20_000


class UnsupervisedLearning:
    """
    Runs all clustering algorithms, evaluates them, and relabels the
    cluster IDs to PHQ-9 severity categories.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.visualizer   = Visualization()
        self.metrics: dict = {"Clustering": {}}
        self.relabeled_df: pd.DataFrame = pd.DataFrame()
        self._optimal_k:    int         = N_PHQ9_CATS
        self._X_scaled:     np.ndarray  = np.empty(0)
        self._embedding_2d: np.ndarray  = np.empty(0)

    # ── Main entry point ─────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> tuple:
        log.info("=" * 60)
        log.info("UNSUPERVISED LEARNING")
        log.info("=" * 60)
        log.info("Input dataframe: %d rows × %d columns", *df.shape)

        exclude = {"phq9_category", "phq9_total"}
        feature_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in exclude
        ]
        X_raw = df[feature_cols].values
        log.info("Feature matrix: %d samples × %d features", *X_raw.shape)

        with MemoryGuard("StandardScaler fit_transform"):
            scaler = StandardScaler()
            self._X_scaled = scaler.fit_transform(X_raw)

        log.info("[Unsupervised] Computing 2-D PCA embedding for scatter plots...")
        with MemoryGuard("PCA 2-D embedding"):
            pca = PCA(n_components=2, random_state=self.random_state)
            self._embedding_2d = pca.fit_transform(self._X_scaled)

        log_memory("After scaling + PCA embedding")

        self.run_kmeans()
        self.run_fuzzy_cmeans()
        self.run_dbscan()
        self.run_subtractive()
        self.run_agglomerative()

        self.relabeled_df = self.relabel(df)
        return self.relabeled_df, self.metrics

    # ── K-Means ──────────────────────────────────────────────────────────────
    def run_kmeans(self) -> None:
        log.info("[Unsupervised] K-Means — Elbow Method (k=%d..%d)", K_MIN, K_MAX)

        k_values, inertias, silhouettes = [], [], []

        for k in range(K_MIN, K_MAX + 1):
            check_memory(threshold_gb=2, label=f"KMeans k={k}")
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(self._X_scaled)
            inertias.append(km.inertia_)
            sil = silhouette_score(
                self._X_scaled, labels,
                sample_size=2000,
                random_state=self.random_state,
            )
            silhouettes.append(sil)
            k_values.append(k)
            log.info("  k=%2d  Inertia=%,.0f  Silhouette=%.4f", k, km.inertia_, sil)

        inertia_arr  = np.array(inertias)
        second_deriv = np.diff(inertia_arr, n=2)
        elbow_idx    = int(np.argmax(second_deriv)) + 1
        self._optimal_k = k_values[elbow_idx]
        log.info("[Unsupervised] Elbow detected at k = %d", self._optimal_k)

        self.visualizer.elbow_plot(k_values, inertias)
        self.visualizer.silhouette_plot(k_values, silhouettes)

        km_final = KMeans(
            n_clusters=self._optimal_k, n_init=10, random_state=self.random_state
        )
        with MemoryGuard(f"KMeans final (k={self._optimal_k})"):
            labels_final = km_final.fit_predict(self._X_scaled)

        self.store_metrics("KMeans", labels_final)
        self.visualizer.cluster_scatter_2d(
            embedding_2d=self._embedding_2d,
            labels=labels_final,
            title=f"K-Means Clustering (k = {self._optimal_k})",
            filename="kmeans_clusters.png",
        )
        log.info(
            "[Unsupervised] KMeans done — Silhouette=%.4f",
            self.metrics["Clustering"]["KMeans"]["Silhouette"],
        )

    # ── Fuzzy C-Means ────────────────────────────────────────────────────────
    def run_fuzzy_cmeans(self) -> None:
        if not _HAS_SKFUZZY:
            log.warning("[Unsupervised] skfuzzy not installed — skipping Fuzzy C-Means")
            return

        log.info(
            "[Unsupervised] Fuzzy C-Means (c = %d)", self._optimal_k
        )

        n = len(self._X_scaled)
        if n > FCM_SAMPLE_THRESHOLD:
            log.info(
                "[Unsupervised] FCM: dataset (%d rows) > threshold (%d) — "
                "fitting on a %d-row sample, assigning the rest by nearest centroid.",
                n, FCM_SAMPLE_THRESHOLD, FCM_MAX_SAMPLES,
            )
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n, FCM_MAX_SAMPLES, replace=False)
            X_fit = self._X_scaled[idx]
        else:
            X_fit = self._X_scaled

        with MemoryGuard(f"Fuzzy C-Means fit (n={len(X_fit)})"):
            cntr, membership, *_ = fuzz.cluster.cmeans(
                data=X_fit.T,
                c=self._optimal_k,
                m=2.0,
                error=0.005,
                maxiter=1000,
                init=None,
                seed=self.random_state,
            )

        # Assign all points to nearest centroid (Euclidean distance)
        dists  = np.linalg.norm(self._X_scaled[:, None] - cntr[None], axis=2)
        labels = np.argmin(dists, axis=1)

        self.store_metrics("FuzzyCMeans", labels)
        self.visualizer.cluster_scatter_2d(
            embedding_2d=self._embedding_2d,
            labels=labels,
            title=f"Fuzzy C-Means Clustering (c = {self._optimal_k})",
            filename="fuzzy_cmeans_clusters.png",
        )
        log.info(
            "[Unsupervised] FuzzyCMeans done — Silhouette=%.4f",
            self.metrics["Clustering"]["FuzzyCMeans"]["Silhouette"],
        )

    # ── DBSCAN ───────────────────────────────────────────────────────────────
    def run_dbscan(self) -> None:
        log.info("[Unsupervised] DBSCAN")

        n = len(self._X_scaled)
        if n > KNN_DBSCAN_CAP:
            log.info(
                "[Unsupervised] DBSCAN: subsampling %d rows for epsilon estimation "
                "(full n=%d)", KNN_DBSCAN_CAP, n,
            )
            rng = np.random.default_rng(self.random_state)
            idx_eps = rng.choice(n, KNN_DBSCAN_CAP, replace=False)
            X_eps   = self._X_scaled[idx_eps]
        else:
            X_eps = self._X_scaled

        with MemoryGuard("DBSCAN k-NN estimation"):
            nbrs = NearestNeighbors(n_neighbors=4).fit(X_eps)
            distances, _ = nbrs.kneighbors(X_eps)

        knn_dists = np.sort(distances[:, -1])[::-1]
        eps = float(np.percentile(knn_dists, 1))
        eps = max(eps, 0.5)
        log.info("[Unsupervised] DBSCAN estimated eps=%.3f", eps)

        with MemoryGuard("DBSCAN fit_predict"):
            db     = DBSCAN(eps=eps, min_samples=5)
            labels = db.fit_predict(self._X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = int((labels == -1).sum())
        log.info(
            "[Unsupervised] DBSCAN — Clusters: %d  Noise: %d", n_clusters, n_noise
        )

        valid_mask = labels != -1
        if n_clusters >= 2 and valid_mask.sum() > n_clusters:
            sil = silhouette_score(
                self._X_scaled[valid_mask], labels[valid_mask],
                sample_size=min(2000, valid_mask.sum()),
                random_state=self.random_state,
            )
            db_score = davies_bouldin_score(
                self._X_scaled[valid_mask], labels[valid_mask]
            )
        else:
            sil, db_score = float("nan"), float("nan")
            log.warning("[Unsupervised] DBSCAN: fewer than 2 valid clusters — metrics N/A")

        self.metrics["Clustering"]["DBSCAN"] = {
            "N_Clusters":     n_clusters,
            "N_Noise":        n_noise,
            "Silhouette":     round(float(sil),      4) if not np.isnan(sil)      else None,
            "Davies_Bouldin": round(float(db_score), 4) if not np.isnan(db_score) else None,
        }

        self.visualizer.cluster_scatter_2d(
            embedding_2d=self._embedding_2d,
            labels=labels,
            title=f"DBSCAN Clustering ({n_clusters} clusters, {n_noise} noise)",
            filename="dbscan_clusters.png",
        )
        log.info(
            "[Unsupervised] DBSCAN done — Silhouette=%s",
            f"{sil:.4f}" if not np.isnan(sil) else "N/A",
        )

    # ── Subtractive Clustering ───────────────────────────────────────────────
    def run_subtractive(self) -> None:
        log.info("[Unsupervised] Subtractive Clustering")

        MAX_SAMPLES = 2_000
        X = self._X_scaled

        if len(X) > MAX_SAMPLES:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), MAX_SAMPLES, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X

        log.info("[Unsupervised] Subtractive: running on %d points", len(X_sub))

        with MemoryGuard("Subtractive Clustering"):
            centers = self.subtractive_clustering(
                X_sub, r_a=0.5, r_b=0.75, eps_upper=0.5, eps_lower=0.15
            )

        n_centers = len(centers)
        log.info("[Unsupervised] Subtractive: %d cluster centers found", n_centers)

        if n_centers == 0:
            labels = np.zeros(len(X), dtype=int)
        else:
            dists  = np.linalg.norm(X[:, None] - np.array(centers)[None], axis=2)
            labels = np.argmin(dists, axis=1)

        self.store_metrics("Subtractive", labels)
        self.visualizer.cluster_scatter_2d(
            embedding_2d=self._embedding_2d,
            labels=labels,
            title=f"Subtractive Clustering ({n_centers} clusters)",
            filename="subtractive_clusters.png",
        )
        log.info(
            "[Unsupervised] Subtractive done — Silhouette=%.4f",
            self.metrics["Clustering"]["Subtractive"]["Silhouette"],
        )

    @staticmethod
    def subtractive_clustering(
        X: np.ndarray,
        r_a: float = 0.5,
        r_b: float = 0.75,
        eps_upper: float = 0.5,
        eps_lower: float = 0.15,
    ) -> list:
        alpha = 4.0 / (r_a ** 2)
        beta  = 4.0 / (r_b ** 2)
        n     = len(X)
        P     = np.zeros(n)

        for i in range(n):
            diffs = X - X[i]
            P[i]  = np.sum(np.exp(-alpha * np.sum(diffs ** 2, axis=1)))

        centers    = []
        P_working  = P.copy()
        P_first    = P.max()

        while True:
            best_idx = int(np.argmax(P_working))
            best_P   = P_working[best_idx]
            ratio    = best_P / P_first

            if ratio >= eps_upper:
                centers.append(X[best_idx].copy())
            elif ratio < eps_lower:
                break
            else:
                d_min = min(
                    np.linalg.norm(X[best_idx] - c) for c in centers
                ) if centers else float("inf")
                if (ratio + d_min / r_a) >= 1.0:
                    centers.append(X[best_idx].copy())
                else:
                    break

            diffs     = X - X[best_idx]
            P_working -= best_P * np.exp(-beta * np.sum(diffs ** 2, axis=1))

            if len(centers) >= 20:
                break

        return centers

    # ── Agglomerative Clustering ─────────────────────────────────────────────
    def run_agglomerative(self) -> None:
        """
        Agglomerative Clustering — memory-safe approach for large datasets.

        The original Ward linkage builds an O(n²) dendrogram in RAM.
        On 1 M rows this means ~4 TB of floats — physically impossible.

        Strategy for large datasets (n > AGG_SAMPLE_THRESHOLD)
        ────────────────────────────────────────────────────────
        1. Draw a random sample of AGG_MAX_SAMPLES rows.
        2. Run full Ward linkage on the sample to find k centroids.
        3. Compute the centroid of each discovered cluster.
        4. Assign every row in the full dataset to its nearest centroid.

        This gives a valid (though approximate) agglomerative partition at
        a memory cost of O(s²) where s = AGG_MAX_SAMPLES (≈ 1.6 GB at 40 k).
        """
        log.info(
            "[Unsupervised] Agglomerative Clustering (k = %d)", self._optimal_k
        )

        n = len(self._X_scaled)

        if n > AGG_SAMPLE_THRESHOLD:
            log.info(
                "[Unsupervised] Agglomerative: dataset (%d rows) > threshold (%d). "
                "Running Ward linkage on %d sampled rows, then assigning full set "
                "by nearest centroid.",
                n, AGG_SAMPLE_THRESHOLD, AGG_MAX_SAMPLES,
            )
            rng = np.random.default_rng(self.random_state)
            idx_sample = rng.choice(n, AGG_MAX_SAMPLES, replace=False)
            X_sample   = self._X_scaled[idx_sample]

            with MemoryGuard(
                f"Agglomerative Ward (sample={AGG_MAX_SAMPLES})", threshold_gb=3
            ):
                agg    = AgglomerativeClustering(n_clusters=self._optimal_k, linkage="ward")
                labels_sample = agg.fit_predict(X_sample)

            # Compute centroids from the sample
            centroids = np.array([
                X_sample[labels_sample == k].mean(axis=0)
                for k in range(self._optimal_k)
            ])

            log.info("[Unsupervised] Agglomerative: assigning %d rows to %d centroids...", n, self._optimal_k)
            # Batch assignment to avoid a single (n × k) distance matrix
            BATCH = 100_000
            labels = np.empty(n, dtype=int)
            for start in range(0, n, BATCH):
                end = min(start + BATCH, n)
                dists = np.linalg.norm(
                    self._X_scaled[start:end, None] - centroids[None], axis=2
                )
                labels[start:end] = np.argmin(dists, axis=1)

        else:
            log.info("[Unsupervised] Agglomerative: full Ward linkage on %d rows", n)
            with MemoryGuard("Agglomerative Ward (full)", threshold_gb=3):
                agg    = AgglomerativeClustering(n_clusters=self._optimal_k, linkage="ward")
                labels = agg.fit_predict(self._X_scaled)

        self.store_metrics("Agglomerative", labels)
        self.visualizer.cluster_scatter_2d(
            embedding_2d=self._embedding_2d,
            labels=labels,
            title=f"Agglomerative Clustering (k = {self._optimal_k}, Ward)",
            filename="agglomerative_clusters.png",
        )
        log.info(
            "[Unsupervised] Agglomerative done — Silhouette=%.4f",
            self.metrics["Clustering"]["Agglomerative"]["Silhouette"],
        )
        log_memory("After Agglomerative")

    # ── Shared metric storage ────────────────────────────────────────────────
    def store_metrics(self, name: str, labels: np.ndarray) -> None:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters >= 2:
            sil = silhouette_score(
                self._X_scaled, labels,
                sample_size=min(2000, len(labels)),
                random_state=self.random_state,
            )
            db  = davies_bouldin_score(self._X_scaled, labels)
        else:
            sil, db = float("nan"), float("nan")
            log.warning("[Unsupervised] %s: < 2 clusters — metrics set to None", name)

        self.metrics["Clustering"][name] = {
            "N_Clusters":     n_clusters,
            "Silhouette":     round(float(sil), 4) if not np.isnan(sil) else None,
            "Davies_Bouldin": round(float(db),  4) if not np.isnan(db)  else None,
        }

    # ── Relabelling ──────────────────────────────────────────────────────────
    def relabel(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("[Unsupervised] Relabelling clusters → PHQ-9 severity categories")

        PHQ9_THRESHOLDS = [4, 9, 14, 19, 27]

        def score_to_category(score: float) -> str:
            for threshold, label in zip(PHQ9_THRESHOLDS, PHQ9_CATEGORIES):
                if score <= threshold:
                    return label
            return PHQ9_CATEGORIES[-1]

        with MemoryGuard("KMeans relabel"):
            km = KMeans(n_clusters=self._optimal_k, n_init=10,
                        random_state=self.random_state)
            cluster_ids = km.fit_predict(self._X_scaled)

        df_out = df.copy()
        df_out["_cluster_id"] = cluster_ids

        cluster_medians = df_out.groupby("_cluster_id")["phq9_total"].median()
        cluster_map     = {
            cid: score_to_category(med)
            for cid, med in cluster_medians.items()
        }

        for cid, cat in sorted(cluster_map.items()):
            log.info(
                "  Cluster %d  →  %-20s  (median phq9_total = %.1f)",
                cid, cat, cluster_medians[cid],
            )

        df_out["cluster_label"] = df_out["_cluster_id"].map(cluster_map)
        df_out.drop(columns=["_cluster_id"], inplace=True)
        log.info("[Unsupervised] Relabelling complete.")
        return df_out
