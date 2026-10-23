#!/usr/bin/env bash
# scripts/sync_policies.sh - Sync latest accepted policy from policies branch to working tree

set -euo pipefail

echo "=== Syncing policies branch to main workspace ==="
git fetch . policies:policies || true
LATEST_POLICY=$(git log policies -n 1 --pretty=format:"%s")
echo "Latest accepted policy commit: $LATEST_POLICY"
