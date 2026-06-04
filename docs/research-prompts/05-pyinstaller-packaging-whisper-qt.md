# Deep Research Prompt 05 — PyInstaller Packaging for Python AI/ML Applications: Whisper, Qt, and Cross-Platform Distribution

## Purpose

Lore must be distributed as a standalone desktop application requiring zero Python knowledge to install. This means PyInstaller packaging. Packaging Python applications that bundle AI models (or download them at first run), use Qt, and depend on binary extensions (CTranslate2, onnxruntime) is notoriously fragile. This prompt requests comprehensive technical research covering the current state of PyInstaller for this specific dependency profile, the first-run model download flow, and platform-specific gotchas.

---

## Research Questions

### 1. PyInstaller with faster-whisper / CTranslate2

- What is the **current status of PyInstaller packaging with `faster-whisper`** (CTranslate2 backend)?
  - Are there known issues with PyInstaller not finding CTranslate2 native libraries?
  - What `hiddenimports`, `datas`, and `binaries` declarations are required in the `.spec` file?
  - Are there community-maintained `.spec` file templates or working examples on GitHub?
- Does `CTranslate2` ship its own native `.so`/`.dll` files that need to be explicitly included? How should these be included in a PyInstaller build?
- Does `ctranslate2` require specific CUDA or cuDNN DLLs even in CPU-only mode? If so, which, and how are they bundled?

### 2. PyInstaller with PyQt6

- What are the **known PyQt6 + PyInstaller issues** in 2025–2026?
  - Plugin directories (`qt6_plugins`, `qt6_qml`) — do they need to be explicitly included?
  - Does `PyInstaller` auto-detect Qt platform plugins (xcb on Linux, cocoa on macOS, windows on Windows)?
  - Are there font rendering or icon theme issues on Linux that need to be worked around?
- Is **`briefcase`** (BeeWare) a better alternative to PyInstaller for PyQt6 apps? What are the trade-offs?
- Is **`cx_Freeze`** worth considering instead of PyInstaller for this stack?

### 3. silero-VAD and onnxruntime

- `silero-VAD` (via PyTorch) or silero-VAD via `onnxruntime` — which is easier to bundle with PyInstaller?
- What are the onnxruntime-specific PyInstaller requirements (`hiddenimports`, providers)?
- If silero-VAD is used via `torch`, what is the PyInstaller overhead of bundling PyTorch? Is the `cpu`-only torch wheel significantly smaller than the default?

### 4. First-Run Model Download Flow

- The scope specifies that Whisper models are downloaded at first run, not bundled. What is the **recommended UX pattern** for a first-run model download in a PyInstaller desktop app?
  - Display a progress bar in a dedicated "first launch" window?
  - Download in the background while showing the main window?
  - Require the user to explicitly choose a model before downloading?
- How should the **download be implemented** — using `faster-whisper`'s built-in download mechanism, or a manual `requests`/`httpx` download with progress reporting?
- Where should downloaded models be stored in a PyInstaller application?
  - `%APPDATA%\Lore\models` on Windows?
  - `~/.local/share/lore/models` on Linux?
  - `~/Library/Application Support/Lore/models` on macOS?
  - How does `platformdirs` (the `user_data_dir()` function) handle this cross-platform?
- What happens if the **download is interrupted**? Partial model files? How should resume be handled?
- Should Lore check for **model updates** on subsequent launches? Or is "download once, never update" acceptable for a local model cache?

### 5. Cross-Platform Build

- What is the **minimum supported OS version** for a PyInstaller app targeting:
  - Windows: Win10 x64? Win11? Does the CTranslate2 MSVC runtime requirement affect this?
  - macOS: Ventura? Monterey? Is an Apple Silicon (arm64) native build required?
  - Linux: Ubuntu 22.04+? What glibc version is required?
- Is **GitHub Actions** the right CI environment for building cross-platform PyInstaller artifacts? Provide the exact runner OS for each target:
  - Windows: `windows-latest` (what version)?
  - macOS: `macos-latest` (Intel or ARM?)
  - Linux: `ubuntu-22.04` or `ubuntu-latest`?
- Can a **single GitHub Actions workflow** produce all three platform builds in parallel? What are the limitations?
- Is **code signing** required for distribution without security warnings?
  - Windows: What happens without a code signing certificate (SmartScreen warning)?
  - macOS: What happens without an Apple Developer certificate + notarisation?
  - Linux: No signing required?

### 6. App Size and Startup Time

- What is the **expected installed app size** for a Lore build (excluding models) containing: PyQt6, faster-whisper (CTranslate2), silero-VAD, ffmpeg subprocess?
- What is the expected **cold startup time** for the main window to appear?
- Is **UPX compression** worth enabling for PyInstaller? Does it conflict with any of these libraries?
- Are there any libraries in this stack that are **known to slow startup** and should be lazy-loaded?

### 7. ffmpeg Bundling

- Lore uses ffmpeg as a subprocess for audio format conversion. Should ffmpeg be:
  a. Bundled inside the PyInstaller package (using `ffmpeg-python` or `imageio-ffmpeg`)?
  b. Required as an external system dependency (user must install separately)?
  c. Downloaded at first run alongside the Whisper model?
- What is the **legal situation** for bundling ffmpeg binaries — are there licence compatibility issues between ffmpeg (LGPL/GPL) and the MIT-licensed Lore?
- The `imageio-ffmpeg` package bundles ffmpeg static binaries. Is this the recommended approach for PyInstaller packaging?

---

## Context

**Project:** Lore — oral history transcription tool  
**Distribution:** PyInstaller standalone executables, no Python required  
**Platforms:** Windows 10+, macOS Ventura+, Ubuntu 22.04+  
**CI:** GitHub Actions for automated builds  
**Target user:** Non-technical archivist who installs from a downloaded .exe / .dmg / .deb

---

## Deliverables Requested

1. Working PyInstaller `.spec` file template for faster-whisper + PyQt6 + silero-VAD
2. List of required `hiddenimports`, `datas`, and `binaries` for each major dependency
3. First-run model download UX pattern with resume support
4. Cross-platform model storage paths using `platformdirs`
5. GitHub Actions workflow structure for three-platform builds
6. Expected app size (ex-model) and startup time
7. ffmpeg bundling recommendation (imageio-ffmpeg vs external)

---

*This research will directly inform `lore.spec`, `lore/src/utils/model_manager.py`, `lore/src/utils/paths.py`, and `.github/workflows/build.yml`.*
