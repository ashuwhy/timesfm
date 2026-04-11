#!/usr/bin/env bash
# Add result checkpoints under Git LFS. Requires: brew install git-lfs && git lfs install
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is not installed. On macOS: brew install git-lfs && git lfs install" >&2
  exit 1
fi
git lfs install
git add results/
echo "Staged results/. Review with: git status && git lfs ls-files"
echo "Then: git commit -m 'Add result checkpoints (Git LFS)' && git push"
