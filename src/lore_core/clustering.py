"""
Speaker clustering using sklearn spectral clustering with GMM-BIC
speaker-count estimation. Reference architecture from the diarize
library (Apache 2.0), reimplemented without the torch dependency.
"""

from typing import Optional
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture


def estimate_num_speakers(embeddings: np.ndarray, max_speakers: int = 8) -> int:
    """
    Estimate the number of speakers using GMM-BIC.

    Fits Gaussian Mixture Models with 1..max_speakers components and
    selects the one with the lowest Bayesian Information Criterion.

    Reference: diarize library GMM-BIC approach (Apache 2.0).
    """
    if len(embeddings) == 1:
        return 1

    best_k = 1
    best_bic = float("inf")

    for k in range(1, min(max_speakers + 1, len(embeddings) + 1)):
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
        gmm.fit(embeddings)
        bic = gmm.bic(embeddings)
        if bic < best_bic:
            best_bic = bic
            best_k = k

    return best_k


class SpeakerClustering:
    """Cluster speaker embeddings into speaker labels."""

    def __init__(self, n_speakers: Optional[int] = None):
        """
        Args:
            n_speakers: Known speaker count. If None, estimate via GMM-BIC.
        """
        self.n_speakers = n_speakers

    def cluster(self, embeddings: np.ndarray) -> list[int]:
        """
        Cluster embeddings and return label indices.

        Args:
            embeddings: (n_segments, embedding_dim) float32 array

        Returns:
            labels: list of ints, same length as n_segments
        """
        n = len(embeddings)
        if n == 0:
            return []
        if n == 1:
            return [0]

        n_clusters = self.n_speakers if self.n_speakers else estimate_num_speakers(embeddings)
        n_clusters = min(n_clusters, n)  # Can't have more clusters than samples

        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            random_state=42,
            affinity="rbf",
            assign_labels="kmeans",
        )
        labels = clusterer.fit_predict(embeddings)
        return labels.tolist()
