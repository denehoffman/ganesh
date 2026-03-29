#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sampler="${1:-aies}"
dim="${2:-4}"
walkers="${3:-32}"
steps="${4:-2000}"

mkdir -p target/memory-profiles
log_file="target/memory-profiles/mcmc-${sampler}-d${dim}-w${walkers}-s${steps}.log"

python3 "${ROOT_DIR}/scripts/profile_with_rss.py" \
  cargo run --release --example profile_mcmc_memory -- \
  "${sampler}" "${dim}" "${walkers}" "${steps}" 2>&1 | tee "${log_file}"

echo
echo "Saved profile log to ${log_file}"
