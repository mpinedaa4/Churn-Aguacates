import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from Visualization.Visualization import Visualization

# PHQ-9 category ordering (for relabelling)
PHQ9_CATEGORIES = ["Minimal", "Mild", "Moderate", "Moderately_Severe", "Severe"]
N_PHQ9_CATS = len(PHQ9_CATEGORIES)

# K-Means search range
K_MIN, K_MAX = 2, 10


class UnsupervisedLearning:
    """
    Runs all clustering algorithms, evaluates them, and (if needed) relabels
    the dataset so the cluster labels align with PHQ-9 severity categories.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state # Reproducibility seed
        self.visualizer = Visualization()
        self.metrics: dict = {"Clustering": {}}
        self.relabeled_df: pd.DataFrame = pd.DataFrame()
        self._optimal_k: int = N_PHQ9_CATS # default; updated by elbow
        self._X_scaled: np.ndarray = np.empty(0) # Scaled featur matrix for clustering
        self._embedding_2d: np.ndarray = np.empty(0) # 2-D PCA embedding for visualisation

    def run(self, df: pd.DataFrame) -> tuple:
        """
        Execute the full unsupervised pipeline.
        """

        print("\n" + "=" * 60)
        print("UNSUPERVISED LEARNING")
        print("=" * 60)

        # Prepare numeric feature matrix (exclude target columns)
        exclude = {"phq9_category", "phq9_total"}
        feature_cols = [c for c in df.select_dtypes(include="number").columns
                        if c not in exclude]
        X_raw = df[feature_cols].values

        # Normalization (standard scaling) for clustering algorithms
        scaler = StandardScaler()
        self._X_scaled = scaler.fit_transform(X_raw)

        # 2-D PCA embedding for cluster scatter visualisations
        pca = PCA(n_components=2, random_state=self.random_state)
        self._embedding_2d = pca.fit_transform(self._X_scaled)

        # Run models
        self.run_kmeans()
        self.run_fuzzy_cmeans()
        self.run_dbscan()
        self.run_subtractive()
        self.run_agglomerative()

        # Relabelling
        self.relabeled_df = self.relabel(df)

        return self.relabeled_df, self.metrics

    # K-Means with Elbow Method
    def run_kmeans(self) -> None:
        print("\n[Unsupervised] K-Means — Elbow Method")

        k_values, inertias, silhouettes = [], [], []

        for k in range(K_MIN, K_MAX + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(self._X_scaled)
            inertias.append(km.inertia_)
            sil = silhouette_score(self._X_scaled, labels, sample_size=1000,
                                   random_state=self.random_state)
            silhouettes.append(sil)
            k_values.append(k)

            print(f"  k={k:2d}  Inertia={km.inertia_:,.0f}  Silhouette={sil:.4f}")

        # Elbow detection using maximum second-derivative of inertia
        inertia_arr = np.array(inertias)
        second_deriv = np.diff(inertia_arr, n=2) # second derivative

        # +1 because the second-derivative shifts index by 2
        elbow_idx = int(np.argmax(second_deriv)) + 1
        self._optimal_k = k_values[elbow_idx] # Optimal K using elbow method
        print(f"\n[Unsupervised] Elbow detected at k = {self._optimal_k}")

        # Visualise
        self.visualizer.elbow_plot(k_values, inertias)
        self.visualizer.silhouette_plot(k_values, silhouettes)

        # Train final K-Means with optimal k
        km_final = KMeans(
            n_clusters=self._optimal_k,
            n_init=10,
            random_state=self.random_state,
        )

        labels_final = km_final.fit_predict(self._X_scaled)
        self.store_metrics("KMeans", labels_final)

        self.visualizer.cluster_scatter_2d(
            embedding_2d=self._embedding_2d,
            labels=labels_final,
            title=f"K-Means Clustering (k = {self._optimal_k})",
            filename="kmeans_clusters.png",
        )

    # Fuzzy C-Means
    def run_fuzzy_cmeans(self) -> None:
        """
        Apply Fuzzy C-Means with the optimal k determined by K-Means.
        Each sample is assigned to the cluster with the highest membership.
        """

        print(f"\n[Unsupervised] Fuzzy C-Means (c = {self._optimal_k})")

        # skfuzzy expects features as columns, samples as rows (transposed)
        X_T = self._X_scaled.T

        cntr, membership, *_ = fuzz.cluster.cmeans(
            data=X_T,
            c=self._optimal_k,
            m=2.0, # fuzziness exponent
            error=0.005,
            maxiter=1000,
            init=None,
            seed=self.random_state,
        )

        # Hard label - cluster with maximum membership
        labels = np.argmax(membership, axis=0)
        self.store_metrics("FuzzyCMeans", labels)

        self.visualizer.cluster_scatter_2d(
            embedding_2d=self._embedding_2d,
            labels=labels,
            title=f"Fuzzy C-Means Clustering (c = {self._optimal_k})",
            filename="fuzzy_cmeans_clusters.png",
        )

        print(f"Silhouette: {self.metrics['Clustering']['FuzzyCMeans']['Silhouette']:.4f}")

    # DBSCAN
    def run_dbscan(self) -> None:
        """
        Apply DBSCAN with automatically estimated epsilon via the k-nearest-
        neighbour distance.
        """

        print("\n[Unsupervised] DBSCAN")

        # Estimate epsilon: 4th-NN distance sorted ascending
        nbrs = NearestNeighbors(n_neighbors=4).fit(self._X_scaled)
        distances, _ = nbrs.kneighbors(self._X_scaled)
        knn_dists    = np.sort(distances[:, -1])[::-1]

        # Use 99th-percentile distance as a robust epsilon
        eps = float(np.percentile(knn_dists, 1))
        eps = max(eps, 0.5)   # lower bound to avoid degenerate clusters

        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(self._X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = int((labels == -1).sum())

        print(f"eps={eps:.3f}  Clusters found: {n_clusters}  Noise: {n_noise}")

        # Evaluate only on non-noise points
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

        self.metrics["Clustering"]["DBSCAN"] = {
            "N_Clusters": n_clusters,
            "N_Noise": n_noise,
            "Silhouette": round(float(sil), 4)      if not np.isnan(sil)      else None,
            "Davies_Bouldin": round(float(db_score), 4) if not np.isnan(db_score) else None,
        }

        print(f"  Silhouette: {sil:.4f}" if not np.isnan(sil) else "  Silhouette: N/A")

        self.visualizer.cluster_scatter_2d(
            embedding_2d=self._embedding_2d,
            labels=labels,
            title=f"DBSCAN Clustering ({n_clusters} clusters, {n_noise} noise)",
            filename="dbscan_clusters.png",
        )

    # Subtractive Clustering
    def run_subtractive(self) -> None:
        """
        Apply the Subtractive Clustering algorithm.

        This algorithm estimates cluster centers by computing a potential
        function for each data point.  The point with the highest potential
        becomes the first center; subsequent centers are identified after
        subtracting the influence of existing centers.
        """

        print("\n[Unsupervised] Subtractive Clustering")

        MAX_SAMPLES = 2000
        X = self._X_scaled

        if len(X) > MAX_SAMPLES:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), MAX_SAMPLES, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X
            idx = np.arange(len(X))

        centers = self.subtractive_clustering(
            X_sub, r_a=0.5, r_b=0.75, eps_upper=0.5, eps_lower=0.15
        )
        n_centers = len(centers)

        print(f"  Cluster centers found: {n_centers}")

        # Assign each sample (full dataset) to nearest center
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

        print(f"Silhouette: {self.metrics['Clustering']['Subtractive']['Silhouette']:.4f}")

    @staticmethod
    def subtractive_clustering(
        X: np.ndarray,
        r_a: float = 0.5,
        r_b: float = 0.75,
        eps_upper: float = 0.5,
        eps_lower: float = 0.15,
    ) -> list:
        """
         Subtractive Clustering algorithm.
        """

        alpha = 4.0 / (r_a ** 2)
        beta  = 4.0 / (r_b ** 2)

        # Compute initial potential for each point
        n = len(X)
        P = np.zeros(n)

        for i in range(n):
            diffs = X - X[i]
            P[i]  = np.sum(np.exp(-alpha * np.sum(diffs ** 2, axis=1)))

        centers = []
        P_working = P.copy()
        P_first = P.max()

        while True:
            best_idx = int(np.argmax(P_working))
            best_P = P_working[best_idx]

            ratio = best_P / P_first
            if ratio >= eps_upper:
                # Unconditionally accept
                centers.append(X[best_idx].copy())
            elif ratio < eps_lower:
                # Stop
                break
            else:
                # Mountain test: accept only if far from existing centers
                d_min = min(
                    np.linalg.norm(X[best_idx] - c) for c in centers
                ) if centers else float("inf")
                if (ratio + d_min / r_a) >= 1.0:
                    centers.append(X[best_idx].copy())
                else:
                    break

            # Suppress the neighbourhood of the accepted centre
            diffs = X - X[best_idx]
            P_working -= best_P * np.exp(-beta * np.sum(diffs ** 2, axis=1))

            # Safety limit
            if len(centers) >= 20:
                break

        return centers

    # Agglomerative Clustering
    def run_agglomerative(self) -> None:
        """
        Apply Agglomerative Clustering
        """

        print(f"\n[Unsupervised] Agglomerative Clustering (k = {self._optimal_k})")

        agg = AgglomerativeClustering(
            n_clusters=self._optimal_k,
            linkage="ward",
        )

        labels = agg.fit_predict(self._X_scaled)
        self.store_metrics("Agglomerative", labels)

        self.visualizer.cluster_scatter_2d(
            embedding_2d=self._embedding_2d,
            labels=labels,
            title=f"Agglomerative Clustering (k = {self._optimal_k}, Ward)",
            filename="agglomerative_clusters.png",
        )
        print(f"  Silhouette: {self.metrics['Clustering']['Agglomerative']['Silhouette']:.4f}")

    # Shared metric
    def store_metrics(self, name: str, labels: np.ndarray) -> None:
        """
        Compute Silhouette and Davies-Bouldin scores and store them.
        """

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

        self.metrics["Clustering"][name] = {
            "N_Clusters": n_clusters,
            "Silhouette": round(float(sil), 4) if not np.isnan(sil) else None,
            "Davies_Bouldin": round(float(db),  4) if not np.isnan(db)  else None,
        }

    # Relabelling
    def relabel(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map integer cluster IDs to PHQ-9 severity labels.

        If the optimal k equals the number of PHQ-9 categories (5) each
        cluster is mapped to the category whose median phq9_total falls
        within the corresponding severity band. Otherwise, each cluster
        is assigned the most frequent PHQ-9 category among its members.
        """

        print("\n[Unsupervised] Relabelling clusters → PHQ-9 severity categories")

        # Re-run K-Means with optimal k to get per-row labels on the full set
        km = KMeans(n_clusters=self._optimal_k, n_init=10,
                    random_state=self.random_state)
        cluster_ids = km.fit_predict(self._X_scaled)

        df_out = df.copy()
        df_out["_cluster_id"] = cluster_ids

        # Map each cluster to a PHQ-9 category via median phq9_total
        PHQ9_THRESHOLDS = [4, 9, 14, 19, 27]

        def score_to_category(score: float) -> str:
            for threshold, label in zip(PHQ9_THRESHOLDS, PHQ9_CATEGORIES):
                if score <= threshold:
                    return label
            return PHQ9_CATEGORIES[-1]

        cluster_medians = (
            df_out.groupby("_cluster_id")["phq9_total"]
            .median()
        )

        cluster_map = {
            cid: score_to_category(med)
            for cid, med in cluster_medians.items()
        }

        print("  Cluster → Category mapping:")

        for cid, cat in sorted(cluster_map.items()):
            print(f"    Cluster {cid}  →  {cat}  "
                  f"(median phq9_total = {cluster_medians[cid]:.1f})")

        df_out["cluster_label"] = df_out["_cluster_id"].map(cluster_map)
        df_out.drop(columns=["_cluster_id"], inplace=True)

        return df_out
