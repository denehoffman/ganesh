#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sampler="${1:-aies}"
dim="${2:-4}"
walkers="${3:-32}"
steps="${4:-2000}"
extra_args=("${@:5}")

mkdir -p target/memory-profiles
if [[ "${#extra_args[@]}" -gt 0 ]]; then
  suffix="$(printf -- '-%s' "${extra_args[@]}")"
  log_file="target/memory-profiles/mcmc-${sampler}-d${dim}-w${walkers}-s${steps}${suffix}.log"
else
  log_file="target/memory-profiles/mcmc-${sampler}-d${dim}-w${walkers}-s${steps}.log"
fi

python3 "${ROOT_DIR}/scripts/profile_with_rss.py" \
  cargo run --release --example profile_mcmc_memory -- \
  "${sampler}" "${dim}" "${walkers}" "${steps}" "${extra_args[@]}" 2>&1 | tee "${log_file}"

echo
echo "Saved profile log to ${log_file}"
