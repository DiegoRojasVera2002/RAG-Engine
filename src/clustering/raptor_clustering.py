"""
RAPTOR-style automatic clustering for hierarchical document organization.

Based on: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval" (ICLR 2024)
Paper: https://arxiv.org/abs/2401.18059

Generates automatic metadata:
- cluster_id: Hierarchical cluster identifier
- tree_level: Level in hierarchy (0=original, 1=summary, etc.)
- cluster_size: Number of chunks in cluster
- centroid_similarity: Similarity to cluster centroid

No manual field definition required - fully automatic.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClusterMetadata:
    """Automatic cluster metadata for a chunk."""
    cluster_id: str
    tree_level: int
    cluster_size: int
    centroid_similarity: float
    parent_cluster_id: str = None


class RAPTORClustering:
    """
    RAPTOR-style hierarchical clustering without LLM.

    Uses Gaussian Mixture Models for soft clustering,
    generating automatic hierarchical structure.
    """

    def __init__(
        self,
        n_levels: int = 2,
        reduction_factor: int = 10,
        random_state: int = 42
    ):
        """
        Initialize RAPTOR clusterer.

        Args:
            n_levels: Number of hierarchy levels (default: 2)
            reduction_factor: Cluster reduction per level (default: 10)
            random_state: Random seed for reproducibility
        """
        self.n_levels = n_levels
        self.reduction_factor = reduction_factor
        self.random_state = random_state

        logger.info(
            "Initialized RAPTOR clustering",
            extra={
                "n_levels": n_levels,
                "reduction_factor": reduction_factor
            }
        )

    def _optimal_clusters(self, n_items: int, level: int) -> int:
        """
        Calculate optimal number of clusters for given level.

        Args:
            n_items: Number of items to cluster
            level: Current tree level

        Returns:
            Optimal number of clusters
        """
        target = max(
            2,
            min(
                n_items // self.reduction_factor,
                int(np.sqrt(n_items))
            )
        )
        return target

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Cluster embeddings using GMM.

        Args:
            embeddings: Array of shape (n_samples, n_features)
            n_clusters: Number of clusters

        Returns:
            Tuple of (labels, centroids, probabilities)
        """
        if len(embeddings) <= n_clusters:
            labels = np.arange(len(embeddings))
            centroids = embeddings
            probs = [1.0] * len(embeddings)
            return labels, centroids, probs

        try:
            gmm = GaussianMixture(
                n_components=n_clusters,
                random_state=self.random_state,
                covariance_type='tied'
            )
            labels = gmm.fit_predict(embeddings)
            probs = gmm.predict_proba(embeddings).max(axis=1).tolist()

            centroids = np.array([
                embeddings[labels == i].mean(axis=0)
                for i in range(n_clusters)
            ])

            logger.debug(
                "GMM clustering complete",
                extra={
                    "n_samples": len(embeddings),
                    "n_clusters": n_clusters,
                    "unique_labels": len(np.unique(labels))
                }
            )

            return labels, centroids, probs

        except Exception as e:
            logger.warning(
                "GMM failed, falling back to agglomerative",
                extra={"error": str(e)}
            )

            agg = AgglomerativeClustering(n_clusters=n_clusters)
            labels = agg.fit_predict(embeddings)

            centroids = np.array([
                embeddings[labels == i].mean(axis=0)
                for i in range(n_clusters)
            ])

            probs = [1.0] * len(embeddings)

            return labels, centroids, probs

    def _calculate_centroid_similarity(
        self,
        embedding: np.ndarray,
        centroid: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity to centroid.

        Args:
            embedding: Chunk embedding
            centroid: Cluster centroid

        Returns:
            Cosine similarity [0, 1]
        """
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-10)
        norm_cent = centroid / (np.linalg.norm(centroid) + 1e-10)
        similarity = np.dot(norm_emb, norm_cent)
        return float(np.clip(similarity, 0, 1))

    def cluster_hierarchical(
        self,
        embeddings: List[np.ndarray]
    ) -> List[List[ClusterMetadata]]:
        """
        Perform hierarchical clustering on embeddings.

        Args:
            embeddings: List of chunk embeddings

        Returns:
            List of cluster metadata per level
        """
        if not embeddings:
            logger.warning("No embeddings provided for clustering")
            return []

        embeddings_array = np.array(embeddings)
        n_chunks = len(embeddings_array)

        logger.info(
            "Starting hierarchical clustering",
            extra={
                "n_chunks": n_chunks,
                "n_levels": self.n_levels,
                "embedding_dim": embeddings_array.shape[1]
            }
        )

        all_levels_metadata = []
        current_embeddings = embeddings_array
        current_indices = np.arange(n_chunks)

        for level in range(self.n_levels):
            n_clusters = self._optimal_clusters(len(current_embeddings), level)

            logger.info(
                f"Clustering level {level}",
                extra={
                    "n_items": len(current_embeddings),
                    "n_clusters": n_clusters
                }
            )

            labels, centroids, probs = self._cluster_embeddings(
                current_embeddings,
                n_clusters
            )

            level_metadata = []
            cluster_sizes = {}

            for idx in range(len(current_embeddings)):
                cluster_label = int(labels[idx])
                cluster_sizes[cluster_label] = cluster_sizes.get(cluster_label, 0) + 1

            for idx in range(len(current_embeddings)):
                cluster_label = int(labels[idx])
                original_idx = current_indices[idx]

                centroid_sim = self._calculate_centroid_similarity(
                    current_embeddings[idx],
                    centroids[cluster_label]
                )

                metadata = ClusterMetadata(
                    cluster_id=f"L{level}_C{cluster_label}",
                    tree_level=level,
                    cluster_size=cluster_sizes[cluster_label],
                    centroid_similarity=centroid_sim,
                    parent_cluster_id=f"L{level-1}_C{labels[idx]}" if level > 0 else None
                )

                level_metadata.append(metadata)

            all_levels_metadata.append(level_metadata)

            if level < self.n_levels - 1:
                current_embeddings = centroids
                current_indices = np.unique(labels)

        logger.info(
            "Hierarchical clustering complete",
            extra={
                "total_levels": len(all_levels_metadata),
                "chunks_per_level": [len(m) for m in all_levels_metadata]
            }
        )

        return all_levels_metadata


def cluster_documents(
    embeddings: List[np.ndarray],
    n_levels: int = 2,
    reduction_factor: int = 10
) -> List[ClusterMetadata]:
    """
    Convenience function for document clustering.

    Args:
        embeddings: List of chunk embeddings
        n_levels: Number of hierarchy levels
        reduction_factor: Cluster reduction per level

    Returns:
        List of cluster metadata (level 0 only)
    """
    clusterer = RAPTORClustering(
        n_levels=n_levels,
        reduction_factor=reduction_factor
    )

    all_levels = clusterer.cluster_hierarchical(embeddings)

    if not all_levels:
        return []

    return all_levels[0]
