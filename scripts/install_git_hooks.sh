#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

git config core.hooksPath .githooks
chmod +x .githooks/pre-commit

echo "[OK] Git hooks path set to .githooks"
echo "[OK] pre-commit hook is executable"
