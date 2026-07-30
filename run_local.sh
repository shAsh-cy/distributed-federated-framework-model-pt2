#!/usr/bin/env bash
# Run a federated learning experiment locally: one server, N clients.
#
#   ./run_local.sh [config] [num_clients]
#
# Clients need no --cid: a client with no id is assigned the next free shard by
# the server at registration, so starting N of them claims N distinct shards.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
NUM_CLIENTS="${2:-10}"
METRICS="${METRICS_OUT:-results/local_run.json}"

mkdir -p "$(dirname "$METRICS")"

cleanup() {
  echo "Shutting down..."
  kill $(jobs -p) 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting server with $CONFIG (metrics -> $METRICS)"
python -m fl.server --config "$CONFIG" --metrics-out "$METRICS" &
SERVER_PID=$!

# Wait for the port to accept connections rather than sleeping a fixed interval:
# the first TensorFlow import can take several seconds on a cold cache.
PORT="$(python -c "from fl.config import Config; print(Config.from_yaml('$CONFIG').server.port)")"
for _ in $(seq 1 60); do
  if python -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null; then
    break
  fi
  sleep 1
done

echo "Starting $NUM_CLIENTS clients..."
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
  python -m fl.client --config "$CONFIG" --server "127.0.0.1:$PORT" &
done

wait "$SERVER_PID"
echo "Server finished. Metrics written to $METRICS"
