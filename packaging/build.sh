#!/usr/bin/env bash
# Build the standalone `openreco` binary (macOS / Linux). Run from anywhere.
set -e
cd "$(dirname "$0")/.."
echo "Building openreco ..."
python -m PyInstaller packaging/openreco.spec --noconfirm
echo
[ -f dist/openreco ] && echo "Done -> dist/openreco" || echo "Build did not produce dist/openreco — check the output above."
