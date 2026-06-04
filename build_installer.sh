#!/bin/bash
set -e

echo "Building Lore via PyInstaller..."

# 1. Clean previous builds
rm -rf build/ dist/

# 2. Build the app using PyInstaller
# Ensure we are in the venv so PyInstaller picks up the dependencies
source .venv/bin/activate
pyinstaller Lore.spec --clean --noconfirm

echo "Build complete! App is in dist/Lore/"

# 3. Create a lightweight distribution archive
echo "Packaging into Lore_v1.0.tar.gz..."
tar -czf Lore_v1.0.tar.gz -C dist Lore

echo "Done! You can distribute Lore_v1.0.tar.gz."
echo "Bundle size: $(du -sh Lore_v1.0.tar.gz | awk '{print $1}')"
