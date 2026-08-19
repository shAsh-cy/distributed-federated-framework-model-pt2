#!/usr/bin/env bash
#
# Launch the secure-aggregation overhead measurement, and never beside another
# training run.
#
# WHY THIS EXISTS AND NOT JUST `docker run`. Two training containers on this host
# share six cores. The walltime phase quotes seconds per round secure-vs-plain;
# a second batch running at once inflates both figures by the contention factor
# and leaves no trace in the JSON that they overlapped. So this queues rather
# than starts: it waits for every other federated training container to exit,
# takes the shared batch lock for the whole of its own run, and releases it on
# the way out. Start it now and it begins when the machine is free -- it will sit
# behind the running compression/personalization chain until that chain drains.
#
# THE INTERLOCK (identical to run_compression_batch.sh so the batches serialise):
#   1. A poll on `docker ps`: any container whose image matches FL_TRAIN_IMAGES
#      counts as a training run, whoever started it -- catches the chain's
#      containers, which run off fl-dev.
#   2. The lock directory FL_BATCH_LOCK (../.fl-batch.lock), created atomically by
#      mkdir. Any launcher taking the same lock queues behind this one.
#
# USAGE
#   scripts/run_secagg_overhead_batch.sh --check   # verify prerequisites only
#   nohup scripts/run_secagg_overhead_batch.sh >> ../fl-secagg-overhead-launcher.log 2>&1 &
#
# Resumable: secagg_overhead.py writes one JSON per phase and skips a phase whose
# JSON already exists, so a crashed attempt picks up where it stopped.
# FL_MAX_ATTEMPTS bounds retries.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

hostpath() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi
}

NAME="${FL_BATCH_NAME:-fl-secagg-overhead-batch}"
IMAGE="${FL_IMAGE:-fl-dev:latest}"
# Fashion-MNIST is auto-downloaded into the mounted keras cache; no FEMNIST cache
# is needed for this measurement (small_cnn only).
DATA_DIR="${FL_DATA_DIR:-$REPO/../federated-learning-starter/data}"
LOCK="${FL_BATCH_LOCK:-$REPO/../.fl-batch.lock}"
LOG="${FL_BATCH_LOG:-$REPO/docs/_secagg_overhead_batch.log}"
POLL="${FL_POLL_SECONDS:-60}"
MAX_ATTEMPTS="${FL_MAX_ATTEMPTS:-3}"
# Default to all phases; override to run a subset (e.g. FL_BATCH_ARGS="--phases bytes").
BATCH_ARGS="${FL_BATCH_ARGS:---phases bytes masking walltime dropout}"
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
  # Two traps on purpose (see run_compression_batch.sh): EXIT is a pure cleanup;
  # the signal trap must EXIT, or the handler returns and the retry loop starts
  # the next attempt holding no lock.
  trap 'rm -rf "$LOCK"' EXIT
  trap 'log "signalled; stopping"; docker stop "$NAME" >/dev/null 2>&1 || true; exit 130' INT TERM
  log "took $LOCK"
}

preflight() {
  command -v docker >/dev/null 2>&1 || die "docker not on PATH"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found (docker images)"
  [ -f "$REPO/scripts/secagg_overhead.py" ] || die "not the secagg worktree: $REPO"
  [ -d "$DATA_DIR" ] || die "data dir not found: $DATA_DIR (set FL_DATA_DIR)"
  mkdir -p "$DATA_DIR/keras" "$REPO/docs"
  log "repo      $REPO"
  log "image     $IMAGE"
  log "data      $DATA_DIR"
  log "lock      $LOCK"
  log "log       $LOG"
  log "branch    $(env -u MSYS_NO_PATHCONV git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
}

run_batch() {
  # Attached on purpose: this script must outlive the container so the lock is
  # held for the whole run. Detach the SCRIPT with nohup, not the container.
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
    sh -c "python scripts/secagg_overhead.py $BATCH_ARGS >> docs/_secagg_overhead_batch.log 2>&1"
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
  wait_for_quiet  # re-check inside the lock; close the poll/mkdir race

  local attempt=1
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    log "starting secagg overhead, attempt $attempt/$MAX_ATTEMPTS"
    if run_batch; then
      log "secagg overhead finished cleanly"
      log "results: docs/_secagg_overhead_{bytes,masking,walltime,dropout}.json"
      return 0
    fi
    log "attempt $attempt exited non-zero; the batch resumes from its per-phase JSON"
    attempt=$((attempt + 1))
  done
  die "secagg overhead failed $MAX_ATTEMPTS times; see $LOG"
}

main "$@"
