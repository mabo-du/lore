# Pre-Fetch Models — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let field researchers pre-cache all models over a known-good connection so features don't silently fail or hang offline.

**Architecture:** Unified `huggingface_hub.snapshot_download()` path for both Whisper and hub models, driven by a new CLI entry point in `main.py` that parses `--prefetch-models`, `--tier`, `--check`, and `--offline`. The CLI dispatches to new `ModelManager.prefetch()` and `ModelManager.prefetch_status()` methods. `--offline` sets `HF_HUB_OFFLINE=1` for instant cache-miss failure.

**Tech Stack:** Python 3.12+, `huggingface_hub` (existing), `argparse` (stdlib), `tqdm` (already a transitive dep via `huggingface_hub`)

---

### Task 1: ModelManager — WHISPER_REPO_IDS map, prefetch(), prefetch_status()

**Files:**
- Modify: `src/utils/model_manager.py`
- Create: `tests/test_model_manager.py`

**Details:**

1. Add `WHISPER_REPO_IDS` class dict mapping size strings to HF repo IDs (verified against faster-whisper 1.2.1 source):

```python
    # Whisper HF repo IDs — hardcoded to avoid depending on faster-whisper's
    # private _MODELS dict and to bypass its disabled_tqdm wrapper.
    WHISPER_REPO_IDS = {
        "small": "Systran/faster-whisper-small",
        "medium": "Systran/faster-whisper-medium",
        "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    }
```

2. Add a `_TIER_TO_WHISPER` mapping that maps the Settings quality tier strings ("Fast", "Balanced", "Best Quality") to the corresponding size strings, to bridge the gap between Settings UI values and the Whisper model keys.

3. Add `prefetch(quality_tier: str, progress_callback=None)` classmethod:
   - Takes a tier string ("Fast", "Balanced", "Best Quality")
   - For Whisper: calls `snapshot_download(repo_id=WHISPER_REPO_IDS[size], cache_dir=cache_dir, tqdm_class=tqdm, allow_patterns=[...])`
   - For hub models: calls `snapshot_download(repo_id=MODELS[key], cache_dir=cache_dir, tqdm_class=tqdm)` with appropriate allow_patterns
   - Catches and logs errors per model, continues to next (retry on re-run)
   - Uses shared `snapshot_download` import at module level

4. Add `prefetch_status(quality_tier: str) -> dict[str, bool]` classmethod:
   - Returns `{model_name: is_cached}` for all models in the given tier
   - For Whisper: checks if `snapshot_download(..., local_files_only=True)` succeeds
   - For hub models: uses existing `get_model_path()` or same local_files_only check
   - Includes Whisper models in the check even though they don't report through `get_model_path()`

5. Extract `_get_cache_dir(quality_tier)` helper to deduplicate the NER vs general cache dir logic.

**Tests** (`tests/test_model_manager.py`):
```python
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


class TestPrefetchStatus:
    def test_status_returns_dict(self):
        """Returns a dict, not None or empty, even for valid tier."""
        result = ModelManager.prefetch_status("Fast")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_status_keys_are_strings(self):
        result = ModelManager.prefetch_status("Fast")
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, bool)
```

**Commit:** `git add src/utils/model_manager.py tests/test_model_manager.py && git commit -m "feat: add WHISPER_REPO_IDS, prefetch(), and prefetch_status() to ModelManager"`

---

### Task 2: CLI entry point in main.py

**Files:**
- Modify: `src/main.py`

**Details:**

1. Read `src/main.py` to understand current entry structure.

2. Add argument parsing at the top of the CLI path, before launching the GUI:

The current `main()` likely just calls `QApplication` and shows the window. Restructure to:

```python
def main():
    """Lore: Oral History Transcription — CLI entry point."""
    # Check for CLI-only flags before starting Qt
    import sys
    from utils.model_manager import ModelManager
    
    parser = argparse.ArgumentParser(description="Lore: Oral History Transcription")
    parser.add_argument("--prefetch-models", action="store_true",
                        help="Pre-download all AI models for offline use")
    parser.add_argument("--tier", choices=["Fast", "Balanced", "Best Quality"],
                        default=None,
                        help="Model quality tier for pre-fetch (default: from Settings)")
    parser.add_argument("--check", action="store_true",
                        help="Check which models are cached (with --prefetch-models)")
    parser.add_argument("--offline", action="store_true",
                        help="Set HF_HUB_OFFLINE=1 to fail fast on missing models")
    args, remaining = parser.parse_known_args()
    
    # Set offline mode early if requested
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    
    if args.prefetch_models:
        _run_prefetch(args)
        return
    
    # ... existing GUI startup code ...
```

3. Add `_run_prefetch(args)` function:

```python
def _run_prefetch(args):
    """Run model pre-fetch and/or status check."""
    from utils.model_manager import ModelManager
    from pathlib import Path
    import os
    
    # Determine tier: explicit arg > QSettings > Best Quality
    tier = args.tier
    if tier is None:
        try:
            from PyQt6.QtCore import QSettings
            settings = QSettings("HeritageTools", "Lore")
            tier = settings.value("transcription/model_tier", "Best Quality")
        except Exception:
            tier = "Best Quality"
    
    if args.check:
        status = ModelManager.prefetch_status(tier)
        print(f"\nModel cache status for tier: {tier}")
        print(f"{'Model':<30} {'Status':<10}")
        print("-" * 40)
        all_cached = True
        for model, cached in status.items():
            marker = "✓ cached" if cached else "✗ missing"
            if not cached:
                all_cached = False
            print(f"{model:<30} {marker:<10}")
        print(f"\nOverall: {'All cached' if all_cached else 'Some models missing'}")
        if not args.offline:
            print("Tip: run with --offline to use these cached files without network checks")
        return
    
    # Pre-fetch
    print(f"Pre-fetching models for tier: {tier}")
    if args.offline:
        print("Offline mode enabled — won't attempt network downloads")
        print("Missing models will be reported below\n")
    
    ModelManager.prefetch(tier)
    print("\nDone. Run 'lore --prefetch-models --check' to verify.")
```

4. The `argparse.REMAINDER` or `parse_known_args()` ensures remaining args don't confuse PyQt6 if Qt tries to parse them too.

**Commit:** `git add src/main.py && git commit -m "feat: add --prefetch-models, --tier, --check, --offline CLI flags"`

---

### Task 3: USER_GUIDE — Offline Preparation section

**Files:**
- Modify: `USER_GUIDE.md`

**Details:**

Add a new section "Offline Preparation" between sections 3 and 4 (after Configuring Custom Vocabulary, before Transcription & Diarization):

```markdown
## Offline Preparation

Lore downloads AI models on first use. If you're heading to the field
without reliable internet, pre-cache everything beforehand:

```bash
# Pre-fetch all models for the default tier
lore --prefetch-models

# Pre-fetch for a specific model tier
lore --prefetch-models --tier "Best Quality"

# Check what's cached
lore --prefetch-models --check

# Switch to offline mode (fails fast on missing models)
lore --offline
```

Both `--prefetch-models` and normal Lore operation respect `--offline`:
when set, any missing model raises an immediate error instead of hanging
while `huggingface_hub` times out trying to reach the internet.

**Model cache location:** `~/.local/share/heritage-tools/`
```

Update the Table of Contents to include the new section.

**Commit:** `git add USER_GUIDE.md && git commit -m "docs: add Offline Preparation section to USER_GUIDE"`

---

### Task 4: End-to-end verification

**Steps:**

- [ ] Run `python -m pytest tests/test_model_manager.py -v` — all pass
- [ ] Run `python -m pytest tests/test_signal_chain.py tests/test_overlap_*.py tests/test_ohms_export.py -v` — verify no regressions
- [ ] Verify `lore --help` shows the new flags
- [ ] Verify `lore --prefetch-models --check` without having downloaded models shows "✗ missing"
- [ ] Verify `lore --prefetch-models --check --tier Fast` works
- [ ] Verify `lore --prefetch-models --tier "Best Quality"` downloads models with progress bars
- [ ] Take a note of Balanced tier Whisper model size once downloaded
- [ ] Commit any size documentation updates
