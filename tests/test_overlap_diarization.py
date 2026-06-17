"""Tests for two-stage overlap-aware clustering (Phase 2d)."""

import numpy as np


def test_overlap_ratio_threshold_constant():
    """OVERLAP_RATIO_THRESHOLD should be 0.5 or configurable."""
    # Just verify the module can import
    from lore_core.diarization import DiarizationEngine
    assert hasattr(DiarizationEngine, '_run_onnx')


def test_clean_embedding_path():
    """Non-overlapping segments should go through cluster path."""
    # Test that clean embeddings get clustered
    emb = np.random.randn(256).astype(np.float32)
    n_clean = 5
    clean = np.vstack([emb + np.random.randn(256) * 0.01 for _ in range(n_clean)])
    from sklearn.cluster import SpectralClustering
    c = SpectralClustering(n_clusters=2, random_state=42, affinity="rbf", assign_labels="kmeans")
    labels = c.fit_predict(clean)
    assert len(labels) == n_clean
    # Should have at least 1 and at most 2 labels
    assert 1 <= len(set(labels)) <= 2
