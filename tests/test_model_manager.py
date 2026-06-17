"""Tests for ModelManager prefetch and status."""

from unittest.mock import patch, MagicMock
from utils.model_manager import ModelManager


class TestWhisperRepoIDs:
    def test_has_all_three_tiers(self):
        assert set(ModelManager.WHISPER_REPO_IDS.keys()) == {"small", "medium", "turbo"}

    def test_repo_ids_are_strings(self):
        for k, v in ModelManager.WHISPER_REPO_IDS.items():
            assert isinstance(k, str)
            assert isinstance(v, str)
            assert "/" in v

    def test_turbo_matches_expected(self):
        assert ModelManager.WHISPER_REPO_IDS["turbo"] == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"


class TestTierToWhisper:
    def test_maps_all_tiers(self):
        assert ModelManager._TIER_TO_WHISPER["Fast"] == "small"
        assert ModelManager._TIER_TO_WHISPER["Balanced"] == "medium"
        assert ModelManager._TIER_TO_WHISPER["Best Quality"] == "turbo"


class TestPrefetchStatus:
    def test_status_returns_dict_for_fast(self):
        result = ModelManager.prefetch_status("Fast")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_status_returns_dict_for_best(self):
        result = ModelManager.prefetch_status("Best Quality")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_status_keys_are_strings_values_are_bools(self):
        result = ModelManager.prefetch_status("Fast")
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, bool)


class TestPrefetch:
    def test_prefetch_calls_snapshot_download_for_each_model(self):
        """Mock snapshot_download, verify it's called for each model + whisper."""
        with patch("utils.model_manager.snapshot_download") as mock_dl:
            ModelManager.prefetch("Fast")
            # Fast tier: 1 Whisper + 5 hub models
            assert mock_dl.call_count >= 5
