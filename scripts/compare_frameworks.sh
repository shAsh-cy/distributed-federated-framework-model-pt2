#!/usr/bin/env bash
# Mixed-pool vs single-framework comparison over the real gRPC path.
#
#   ./scripts/compare_frameworks.sh [seeds...]     (default: 42 43 44)
#
# For each seed, runs configs/default.yaml twice: once with 10 TensorFlow
# clients (pure), once with 6 TensorFlow + 4 PyTorch clients (mixed). Same
# server, same config, same seed; the only difference is which framework some
# clients train on. Writes results/compare/{pure,mixed}_<seed>.json.
#
# The comparison is honest only per-seed-pair: the server's cohort sampling is
# seeded, so pure_42 and mixed_42 see identical cohort draws; differences are
# local-training dynamics (framework RNG, batching) plus registration order.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/compare

SEEDS=("${@:-42 43 44}")
if [ $# -eq 0 ]; then SEEDS=(42 43 44); fi

run_pool() {
  local seed="$1" tag="$2" tf_clients="$3" torch_clients="$4"
  local cfg="/tmp/compare_${seed}.yaml"
  sed "s/^seed: .*/seed: ${seed}/" configs/default.yaml > "${cfg}"
  local out="results/compare/${tag}_${seed}.json"

  echo "=== ${tag} seed=${seed}: ${tf_clients} tf + ${torch_clients} torch ==="
  python -m fl.server --config "${cfg}" --metrics-out "${out}" &
  local server_pid=$!

  local port
  port="$(python -c "from fl.config import Config; print(Config.from_yaml('${cfg}').server.port)")"
  for _ in $(seq 1 60); do
    if python -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1',${port}))==0 else 1)" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  for _ in $(seq 1 "${tf_clients}"); do
    python -m fl.client --config "${cfg}" --server "127.0.0.1:${port}" --framework tensorflow &
  done
  for _ in $(seq 1 "${torch_clients}"); do
    python -m fl.client --config "${cfg}" --server "127.0.0.1:${port}" --framework torch &
  done

  wait "${server_pid}"
  # Client processes exit on the server's stop signal; reap them.
  wait || true
  echo "=== ${tag} seed=${seed} done -> ${out} ==="
}

for seed in ${SEEDS[@]+"${SEEDS[@]}"}; do
  run_pool "${seed}" pure 10 0
  run_pool "${seed}" mixed 6 4
done

python - <<'PY'
import json
from pathlib import Path

print(f"{'seed':>5} {'pure final':>11} {'mixed final':>12}")
pures, mixes = [], []
for path in sorted(Path("results/compare").glob("pure_*.json")):
    seed = path.stem.split("_")[1]
    pure = json.loads(path.read_text())["rounds"][-1]["accuracy"]
    mixed = json.loads((path.parent / f"mixed_{seed}.json").read_text())["rounds"][-1]["accuracy"]
    pures.append(pure)
    mixes.append(mixed)
    print(f"{seed:>5} {pure:>11.4f} {mixed:>12.4f}")
mean = lambda v: sum(v) / len(v)  # noqa: E731
rng = lambda v: max(v) - min(v)  # noqa: E731
print(f"{'mean':>5} {mean(pures):>11.4f} {mean(mixes):>12.4f}")
print(f"{'rng':>5} {rng(pures):>11.4f} {rng(mixes):>12.4f}")
PY
