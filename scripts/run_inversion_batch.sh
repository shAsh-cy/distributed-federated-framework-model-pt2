#!/usr/bin/env bash
#
# Launch the gradient-inversion demo, and never beside another training run.
#
# The reconstructions are compute-heavy (thousands of cosine-attack iterations
# over a 225k-parameter model, several times), so like every other batch here it
# queues rather than contends: it waits for other federated training containers
# to exit, takes the shared lock, runs, and releases it. Start it now; it begins
# once the compression/personalization chain drains and the lock frees.
#
# Interlock identical to run_secagg_overhead_batch.sh and run_compression_batch.sh
# (docker-ps poll + the ../.fl-batch.lock directory), so all of them serialise.
#
# USAGE
#   scripts/run_inversion_batch.sh --check   # verify prerequisites only
#   nohup scripts/run_inversion_batch.sh >> ../fl-inversion-launcher.log 2>&1 &
#
# Deterministic under --seed and resumable in spirit: re-running overwrites the
# grids in docs/inversion/ from the same seed, so a crashed attempt is simply
# rerun. FL_MAX_ATTEMPTS bounds retries.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

hostpath() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi
}

NAME="${FL_BATCH_NAME:-fl-inversion-batch}"
IMAGE="${FL_IMAGE:-fl-dev:latest}"
DATA_DIR="${FL_DATA_DIR:-$REPO/../federated-learning-starter/data}"
LOCK="${FL_BATCH_LOCK:-$REPO/../.fl-batch.lock}"
LOG="${FL_BATCH_LOG:-$REPO/docs/inversion/_inversion_batch.log}"
POLL="${FL_POLL_SECONDS:-60}"
MAX_ATTEMPTS="${FL_MAX_ATTEMPTS:-3}"
# Fewer iterations or a subset of phases for a quick pass: FL_BATCH_ARGS="--iters 500 --phases batch".
BATCH_ARGS="${FL_BATCH_ARGS:---seed 0 --iters 4000}"
FL_TRAIN_IMAGES="${FL_TRAIN_IMAGES:-^(fl-dev|fl-app|fl-compress|docker-server|docker-client)}"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "ERROR: $*"; exit 1; }

running_trainers() {
  docker ps --format '{{.Names}} {{.Image}}' \
    | awk -v me="$NAME" -v pat="$FL_TRAIN_IMAGES" '$1 != me && $2 ~ pat { print $1 }'
}

wait_for_quiet() {
  while true; do
    local others
    others="$(running_trainers | tr '\n' ' ')"
    [ -z "${others// /}" ] && return 0
    log "waiting for training container(s) to finish: ${others}"
    sleep "$POLL"
  done
}

take_lock() {
  while ! mkdir "$LOCK" 2>/dev/null; do
    log "another batch holds $LOCK; waiting"
    sleep "$POLL"
  done
  printf '%s pid=%s host=%s\n' "$(date -Iseconds)" "$$" "$(hostname)" > "$LOCK/owner"
  trap 'rm -rf "$LOCK"' EXIT
  trap 'log "signalled; stopping"; docker stop "$NAME" >/dev/null 2>&1 || true; exit 130' INT TERM
  log "took $LOCK"
}

preflight() {
  command -v docker >/dev/null 2>&1 || die "docker not on PATH"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found (docker images)"
  [ -f "$REPO/scripts/inversion_demo.py" ] || die "not the secagg worktree: $REPO"
  [ -d "$DATA_DIR" ] || die "data dir not found: $DATA_DIR (set FL_DATA_DIR)"
  mkdir -p "$DATA_DIR/keras" "$REPO/docs/inversion"
  log "repo      $REPO"
  log "image     $IMAGE"
  log "data      $DATA_DIR"
  log "lock      $LOCK"
  log "log       $LOG"
  log "branch    $(env -u MSYS_NO_PATHCONV git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
}

run_batch() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  MSYS_NO_PATHCONV=1 docker run \
    --name "$NAME" \
    -v "$(hostpath "$REPO")":/app \
    -v "$(hostpath "$DATA_DIR")":/app/data \
    -v "$(hostpath "$DATA_DIR/keras")":/root/.keras \
    -w /app \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
    -e PYTHONPATH=/app \
    "$IMAGE" \
    sh -c "python scripts/inversion_demo.py --out-dir docs/inversion $BATCH_ARGS >> docs/inversion/_inversion_batch.log 2>&1"
}

main() {
  if [ "${1:-}" = "--check" ]; then
    preflight
    local others
    others="$(running_trainers | tr '\n' ' ')"
    log "training containers running now: ${others:-none}"
    [ -d "$LOCK" ] && log "lock is currently held by: $(cat "$LOCK/owner" 2>/dev/null || echo '?')"
    log "check passed; nothing launched"
    return 0
  fi

  preflight
  wait_for_quiet
  take_lock
  wait_for_quiet

  local attempt=1
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    log "starting inversion demo, attempt $attempt/$MAX_ATTEMPTS"
    if run_batch; then
      log "inversion demo finished cleanly"
      log "grids: docs/inversion/*.png, summary docs/inversion/summary.json"
      return 0
    fi
    log "attempt $attempt exited non-zero; rerun from the same seed"
    attempt=$((attempt + 1))
  done
  die "inversion demo failed $MAX_ATTEMPTS times; see $LOG"
}

main "$@"
