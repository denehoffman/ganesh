#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dim="${1:-200}"

mkdir -p target/memory-profiles
log_file="target/memory-profiles/hessian-d${dim}.log"

python3 "${ROOT_DIR}/scripts/profile_with_rss.py" \
  cargo run --release --example profile_hessian_memory -- \
  "${dim}" 2>&1 | tee "${log_file}"

echo
echo "Saved profile log to ${log_file}"
