#!/usr/bin/env bash
# Reproduce every recorded result: the no-DP baseline and both DP settings.
#
#   ./scripts/run_all_experiments.sh
#
# All three configs share seed, model, partition and round count, so the only
# difference between them is the privacy block. Writes results/*.json.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results

run() {
  local config="$1" out="$2"
  echo "===== $config -> results/$out.json ====="
  python scripts/run_experiment.py --config "configs/$config.yaml" --metrics-out "results/$out.json"
}

run default     no_dp
run dp_moderate dp_moderate
run dp_high     dp_high

echo "===== ALL EXPERIMENTS DONE ====="
python scripts/summarise_results.py
