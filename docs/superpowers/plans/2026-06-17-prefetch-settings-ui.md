# Settings UI Toggle — Implementation Plan

## What to change

**File:** `src/ui/settings_dialog.py`

Add a "Offline & Pre-fetch" group box between the Diarization group and the buttons:

```python
        # Offline / Pre-fetch Group
        offline_group = QGroupBox("Offline & Pre-fetch")
        offline_layout = QVBoxLayout(offline_group)

        self.offline_checkbox = QCheckBox(
            "Offline Mode (fail fast on missing models, no network checks)"
        )

        self.prefetch_btn = QPushButton("Download All Models")
        self.prefetch_btn.clicked.connect(self._run_prefetch)
        self.prefetch_progress = QLabel("")

        offline_layout.addWidget(self.offline_checkbox)
        offline_layout.addWidget(self.prefetch_btn)
        offline_layout.addWidget(self.prefetch_progress)

        layout.addWidget(offline_group)
```

Add `QCheckBox` to the imports. Add `_run_prefetch(self)` method that:
1. Gets the current tier from the combo box
2. Calls `ModelManager.prefetch(tier)` in a background thread or blocking call
3. Shows "Downloading..." status via the progress label
4. Shows "Done." or error on completion

Add `load_settings` to read `offline/enabled` bool from QSettings.
Add `save_settings` to persist the checkbox state.

Add `from utils.model_manager import ModelManager` at the top.

**Commit and merge** — no new tests needed for a UI-only change.
