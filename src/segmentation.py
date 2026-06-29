"""
Customer segmentation module for CLV prediction system.
Handles RFM segmentation and clustering analysis.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class CustomerSegmenter:
    """
    Handles customer segmentation using RFM analysis and clustering.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.kmeans_model = None
        self.optimal_clusters = None

    def perform_rfm_segmentation(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform rule-based RFM segmentation.

        Args:
            rfm_df: RFM DataFrame with Recency, Frequency, Monetary columns

        Returns:
            DataFrame with RFM scores and segments
        """
        logger.info("Performing RFM segmentation")

        df = rfm_df.copy()

        # Create quartiles for each RFM metric
        df['R_quartile'] = pd.qcut(df['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop')
        df['F_quartile'] = pd.qcut(df['Frequency'], 4, labels=[1, 2, 3, 4], duplicates='drop')
        df['M_quartile'] = pd.qcut(df['Monetary'], 4, labels=[1, 2, 3, 4], duplicates='drop')

        # Combine into RFM score
        df['RFM_Score'] = df['R_quartile'].astype(str) + df['F_quartile'].astype(str) + df['M_quartile'].astype(str)

        # Define segments based on RFM scores
        df['RFM_Segment'] = df['RFM_Score'].apply(self._assign_rfm_segment)

        logger.info(f"Created RFM segments for {len(df)} customers")
        return df

    def _assign_rfm_segment(self, score: str) -> str:
        """
        Assign segment name based on RFM score.

        Args:
            score: 3-digit RFM score string

        Returns:
            Segment name
        """
        r, f, m = int(score[0]), int(score[1]), int(score[2])

        if r >= 3 and f >= 3 and m >= 3:
            return 'Loyal'
        elif r >= 3 and f >= 2:
            return 'Promising'
        elif r >= 2 and f >= 2:
            return 'Potential'
        elif r >= 2 or f >= 2 or m >= 2:
            return 'Sleep'
        else:
            return 'Dormant'

    def find_optimal_clusters(self, X_scaled: pd.DataFrame, max_clusters: int = 10) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette score.

        Args:
            X_scaled: Scaled feature DataFrame
            max_clusters: Maximum number of clusters to test

        Returns:
            Optimal number of clusters
        """
        logger.info("Finding optimal number of clusters")

        inertias = []
        silhouette_scores = []

        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

            if k > 1:
                score = silhouette_score(X_scaled, kmeans.labels_)
                silhouette_scores.append(score)

        # Find elbow point (simplified approach)
        # Using the point where inertia decrease slows down
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        elbow_point = np.argmin(diffs2) + 2  # +2 because of the double diff

        self.optimal_clusters = min(elbow_point, 5)  # Cap at 5 clusters for interpretability
        logger.info(f"Optimal clusters determined: {self.optimal_clusters}")
        return self.optimal_clusters

    def perform_clustering(self, X_scaled: pd.DataFrame, n_clusters: Optional[int] = None) -> pd.DataFrame:
        """
        Perform K-means clustering on scaled RFM data.

        Args:
            X_scaled: Scaled RFM features
            n_clusters: Number of clusters (if None, uses optimal_clusters)

        Returns:
            DataFrame with cluster assignments
        """
        n_clusters = n_clusters or self.optimal_clusters or 4

        logger.info(f"Performing K-means clustering with {n_clusters} clusters")

        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        clusters = self.kmeans_model.fit_predict(X_scaled)
        cluster_df = pd.DataFrame({
            'CustomerID': X_scaled.index,
            'Cluster': clusters
        })

        # Order clusters by mean monetary value
        cluster_df = self._order_clusters_by_value(X_scaled, cluster_df)

        logger.info("Clustering completed")
        return cluster_df

    def _order_clusters_by_value(self, X_scaled: pd.DataFrame, cluster_df: pd.DataFrame) -> pd.DataFrame:
        """
        Order clusters by their average monetary value for better interpretation.

        Args:
            X_scaled: Scaled features
            cluster_df: DataFrame with cluster assignments

        Returns:
            DataFrame with ordered cluster labels
        """
        # Calculate mean monetary value per cluster
        cluster_values = X_scaled.assign(Cluster=cluster_df['Cluster']).groupby('Cluster')['Monetary_log'].mean().sort_values()

        # Create mapping to reorder clusters (0=lowest value, n-1=highest value)
        cluster_mapping = {old: new for new, old in enumerate(cluster_values.index)}

        cluster_df['Cluster'] = cluster_df['Cluster'].map(cluster_mapping)
        return cluster_df

    def get_cluster_characteristics(self, rfm_df: pd.DataFrame, cluster_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get characteristics of each cluster.

        Args:
            rfm_df: RFM DataFrame
            cluster_df: DataFrame with cluster assignments

        Returns:
            DataFrame with cluster statistics
        """
        combined = rfm_df.join(cluster_df.set_index('CustomerID'), how='inner')

        cluster_stats = combined.groupby('Cluster').agg({
            'Recency': ['mean', 'median', 'count'],
            'Frequency': ['mean', 'median'],
            'Monetary': ['mean', 'median', 'sum']
        }).round(2)

        # Flatten column names
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]

        return cluster_stats

    def plot_clusters(self, X_scaled: pd.DataFrame, cluster_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create visualization of customer clusters.

        Args:
            X_scaled: Scaled RFM features
            cluster_df: DataFrame with cluster assignments
            save_path: Optional path to save the plot
        """
        data = X_scaled.copy()
        data['Cluster'] = cluster_df['Cluster']

        # Create pairplot
        g = sns.pairplot(data, hue='Cluster', palette='viridis', diag_kind='kde')
        g.fig.suptitle('Customer Clusters - RFM Analysis', y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster plot saved to {save_path}")
        else:
            plt.show()
