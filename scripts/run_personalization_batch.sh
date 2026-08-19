#!/usr/bin/env bash
#
# Launch the personalization batch, and never beside another training run.
#
# WHY THIS EXISTS AND NOT JUST `docker run`. Two training containers on this host
# share six cores. docs/personalization.md quotes wall-clock seconds per round,
# and two batches running at once inflates both figures by roughly the contention
# factor while leaving no trace in either JSON that they overlapped. So this
# script queues rather than starts: it waits for every other federated training
# container to exit, takes a lock for the whole of its own run, and releases it
# on the way out.
#
# It is therefore safe to launch this WHILE a phase is still running -- it will
# sit and wait. That is the intended use: start it now, it begins when the
# machine is free.
#
# THE INTERLOCK. This is deliberately the same two-part interlock
# scripts/run_compression_batch.sh defines on feat/compression, and the defaults
# below are byte-identical to that script's so the two queue behind each other
# rather than racing into the same window:
#   1. A poll on `docker ps`: any running container whose image matches
#      FL_TRAIN_IMAGES counts as a training run, whoever started it. This is what
#      catches containers launched by a plain `docker run` -- including the
#      compression batch, and including whatever is running now.
#   2. A lock directory (FL_BATCH_LOCK, default ../.fl-batch.lock), created
#      atomically by mkdir and held until this script's container exits.
#
# It is a near-copy rather than a shared file because of branch topology, not
# preference: run_compression_batch.sh lives on feat/compression, this branch is
# cut from main, and neither is merged. When both land, the two should collapse
# into one parameterised launcher; until then the coupling that matters is the
# LOCK PATH and the IMAGE PATTERN, and those are the two lines to keep in step.
#
# USAGE
#   scripts/run_personalization_batch.sh --check   # verify prerequisites, launch nothing
#   nohup scripts/run_personalization_batch.sh >> ../fl-personal-launcher.log 2>&1 &
#
# The batch itself is resumable -- phase A checkpoints each completed run as it
# finishes, and a phase whose JSON exists is skipped -- so a crashed attempt is
# retried in place and picks up where it stopped. FL_MAX_ATTEMPTS bounds that; it
# is not a loop that can spin, because each attempt either makes progress or
# fails on the same run twice.

set -euo pipefail

# Git Bash rewrites container-side paths like /app into Windows paths before
# Docker sees them, so `docker run` is invoked with MSYS_NO_PATHCONV=1. NOT
# exported for the whole script: git is itself a native Windows binary here and
# needs the opposite -- with the variable set it cannot resolve `-C /c/...` and
# reports every path as missing.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

# Host paths must reach Docker in a form the daemon understands. On MSYS/Cygwin
# that means C:/... rather than /c/...; elsewhere the path is already right.
hostpath() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi
}

NAME="${FL_BATCH_NAME:-fl-personalization-batch}"
IMAGE="${FL_IMAGE:-fl-dev:latest}"
# FEMNIST's packed cache is gitignored and lives in whichever tree prepared it,
# which is not this worktree. Point FL_DATA_DIR at that tree's data/.
DATA_DIR="${FL_DATA_DIR:-$REPO/../federated-learning-starter/data}"
LOCK="${FL_BATCH_LOCK:-$REPO/../.fl-batch.lock}"
LOG="${FL_BATCH_LOG:-$REPO/docs/_personalization.log}"
POLL="${FL_POLL_SECONDS:-60}"
MAX_ATTEMPTS="${FL_MAX_ATTEMPTS:-3}"
# Arguments passed through to the batch. The default runs phase B FIRST: it is
# the 45-minute Fashion phase against the five-hour FEMNIST one, and it is also
# the cleaner signal (Fashion at alpha=0.1 is pure label skew, which is what a
# local head can absorb). If something is wrong with the harness at full scale it
# surfaces in 45 minutes rather than five hours, and because each phase skips a
# JSON that already exists, a rerun picks up at A. Override to exercise the
# launch path alone: FL_BATCH_ARGS="--only B". The zero-cost check is `--check`,
# which launches nothing at all.
BATCH_ARGS="${FL_BATCH_ARGS:---phases B,A}"
# Anything running off one of these images is a federated training container.
# Keep in step with run_compression_batch.sh: a pattern that misses the sibling's
# image would let both run at once while each logged that the host was quiet.
FL_TRAIN_IMAGES="${FL_TRAIN_IMAGES:-^(fl-dev|fl-app|fl-compress|docker-server|docker-client)}"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "ERROR: $*"; exit 1; }

# -- who else is training ----------------------------------------------------

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

# -- the lock ----------------------------------------------------------------

take_lock() {
  while ! mkdir "$LOCK" 2>/dev/null; do
    log "another batch holds $LOCK; waiting"
    sleep "$POLL"
  done
  printf '%s pid=%s host=%s\n' "$(date -Iseconds)" "$$" "$(hostname)" > "$LOCK/owner"
  # A bare `trap ... TERM` releases the lock and then CARRIES ON: the handler
  # returns, the retry loop sees the killed `docker run` as a crash, and the
  # launcher starts the next attempt holding no lock at all. Signals must stop
  # the launcher, so the signal traps exit; only EXIT is a pure cleanup.
  # Ported from run_compression_batch.sh (f6ec594), where it was found stopping
  # a live run. This launcher was copied from that script one commit earlier and
  # inherited the bug; an overnight run is exactly where it would have bitten.
  trap 'rm -rf "$LOCK"' EXIT
  trap 'log "signalled; stopping"; docker stop "$NAME" >/dev/null 2>&1 || true; exit 130' INT TERM
  log "took $LOCK"
}

# -- prerequisites -----------------------------------------------------------

preflight() {
  command -v docker >/dev/null 2>&1 || die "docker not on PATH"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found (docker images)"
  [ -f "$REPO/scripts/personalization_batch.py" ] || die "not a personalization worktree: $REPO"
  [ -d "$DATA_DIR" ] || die "data dir not found: $DATA_DIR (set FL_DATA_DIR)"
  [ -f "$DATA_DIR/femnist/femnist62.npz" ] || die \
    "FEMNIST cache missing at $DATA_DIR/femnist/femnist62.npz. Phase A needs it; run
    python scripts/prepare_femnist.py
  in a tree that has it, or point FL_DATA_DIR at one that does."
  mkdir -p "$DATA_DIR/keras" "$REPO/docs"
  log "repo      $REPO"
  log "image     $IMAGE"
  log "data      $DATA_DIR"
  log "lock      $LOCK"
  log "log       $LOG"
  # `env -u` so the branch still resolves when a caller exported MSYS_NO_PATHCONV
  # themselves: git is a native Windows binary here and cannot resolve
  # `-C /c/...` with that set, which would silently report '?'.
  log "branch    $(env -u MSYS_NO_PATHCONV git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
}

# -- the run -----------------------------------------------------------------

run_batch() {
  # Attached on purpose: this script must outlive the container so the lock is
  # held for the whole run and released when it ends. Detach the SCRIPT with
  # nohup, not the container.
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
    sh -c "python scripts/personalization_batch.py $BATCH_ARGS >> docs/_personalization.log 2>&1"
}

main() {
  if [ "${1:-}" = "--check" ]; then
    preflight
    local others
    others="$(running_trainers | tr '\n' ' ')"
    log "training containers running now: ${others:-none}"
    if [ -d "$LOCK" ]; then
      log "lock HELD by: $(cat "$LOCK/owner" 2>/dev/null || echo '?')"
    else
      log "lock free"
    fi
    # Whether THIS batch is running is a separate question from who holds the
    # lock, and the one a waiting launcher is usually being asked. Report it
    # directly rather than leaving it to be inferred from the container list.
    if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
      log "this batch ($NAME): RUNNING"
    elif pgrep -f "run_personalization_batch.sh" >/dev/null 2>&1; then
      log "this batch ($NAME): QUEUED -- launcher process alive, no container yet"
    else
      log "this batch ($NAME): not running and no launcher process found"
    fi
    log "check passed; nothing launched"
    return 0
  fi

  preflight
  wait_for_quiet
  take_lock
  # Between the poll and the lock another launcher could have started. Re-check
  # once inside the lock; anything that also takes the lock is already excluded.
  wait_for_quiet

  local attempt=1
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    log "starting personalization batch, attempt $attempt/$MAX_ATTEMPTS"
    if run_batch; then
      log "personalization batch finished cleanly"
      log "results: docs/_personalization_a.json, docs/_personalization_b.json"
      return 0
    fi
    log "attempt $attempt exited non-zero; the batch resumes from its checkpoints"
    attempt=$((attempt + 1))
  done
  die "personalization batch failed $MAX_ATTEMPTS times; see $LOG"
}

main "$@"
