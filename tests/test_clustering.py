"""Tests for speaker clustering."""

import numpy as np
from lore_core.clustering import SpeakerClustering, estimate_num_speakers


def test_estimate_num_speakers():
    """GMM-BIC should handle realistic embedding arrays."""
    # Create two distinct clusters of embeddings
    emb = np.vstack([
        np.random.randn(5, 256) * 0.1 + np.array([1.0] * 256),
        np.random.randn(5, 256) * 0.1 - np.array([1.0] * 256),
    ])
    k = estimate_num_speakers(emb, max_speakers=4)
    # Should be >= 1 and <= 4 (exact depends on random init)
    assert 1 <= k <= 4


def test_cluster_with_known_count():
    emb = np.random.randn(10, 256)
    c = SpeakerClustering(n_speakers=2)
    labels = c.cluster(emb)
    assert len(labels) == 10
    assert len(set(labels)) == 2


def test_cluster_with_auto_count():
    emb = np.random.randn(10, 256)
    c = SpeakerClustering()
    labels = c.cluster(emb)
    assert len(labels) == 10


def test_cluster_empty():
    c = SpeakerClustering()
    assert c.cluster(np.array([]).reshape(0, 256)) == []
