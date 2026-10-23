#!/usr/bin/env bash
# scripts/bootstrap.sh - Set up environment with uv

set -euo pipefail

echo "=== Bootstrapping StrikeShield environment ==="
uv sync --extra dev
echo "=== Environment setup complete ==="
