import sys
from pathlib import Path

# Ensure src is in PYTHONPATH when running directly
sys.path.insert(0, str(Path(__file__).parent))

from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow


def _run_prefetch(args):
    """Run model pre-fetch and/or status check."""
    from utils.model_manager import ModelManager

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
        print()
        if all_cached:
            print("All models cached. You can use --offline to skip network checks.")
        else:
            print("Some models are missing. Run without --check to download them.")
            if args.offline:
                print("Note: --offline prevents downloads — cache check only.")
        return

    # Pre-fetch
    print(f"\n=== Pre-fetching models for tier: {tier} ===\n")
    if args.offline:
        print("Offline mode is set — will check cache only, no downloads.\n")
        return

    ModelManager.prefetch(tier)
    print("\nDone. Run 'lore --prefetch-models --check' to verify.")


def main():
    """Lore: Oral History Transcription — CLI entry point."""
    # Check for CLI-only flags before starting Qt
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Lore: Oral History Transcription")
    parser.add_argument(
        "--prefetch-models",
        action="store_true",
        help="Pre-download all AI models for offline use",
    )
    parser.add_argument(
        "--tier",
        choices=["Fast", "Balanced", "Best Quality"],
        default=None,
        help="Model quality tier for pre-fetch (default: from Settings)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check which models are cached (requires --prefetch-models)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 to fail fast on missing models, no downloads",
    )
    args, remaining = parser.parse_known_args()

    # Set offline mode early if requested
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    if args.prefetch_models:
        _run_prefetch(args)
        return

    app = QApplication(sys.argv)

    # Configure global app settings
    app.setApplicationName("Lore")
    app.setOrganizationName("Digital Heritage Lab")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
