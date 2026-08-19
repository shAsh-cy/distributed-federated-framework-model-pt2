"""The secure-aggregation server: the live gRPC path for pairwise masking.

:class:`SecureFederatedServer` subclasses :class:`fl.server.FederatedServer` and
reuses all of its registration, cohort sampling, deadline, evaluation and
metrics machinery. It replaces only the round body: instead of collecting
plaintext ``ModelWeights`` through ``SubmitUpdate``, it runs the Bonawitz et al.
secure round over the V3 RPCs, so the server sums masked words and never sees an
individual update.

Every phase delegates to the transport-agnostic, locally-tested orchestration in
:mod:`fl.secure_round` (:class:`ServerSession`) and the crypto in
:mod:`fl.secure_aggregation`. This module is the marshalling between those and
gRPC — it deliberately holds no protocol arithmetic of its own.

THE DP INCOMPATIBILITY, restated where it is enforced: this path is no-DP by
construction. The TFF DP aggregator clips and noises centrally, after seeing
each update — exactly what masking prevents — so a secure server carries no DP
aggregator and computes the plain sample-count-weighted mean. Composing the two
needs distributed DP (client-side clipping, share-split noise); see
docs/architecture.md and the Roadmap. :func:`build_secure_server` refuses a
config with ``privacy.enabled`` rather than silently ignoring it.

Phases of one secure round (server-driven; clients poll):

1. SUBMIT   the server publishes the cohort roster and global weights; each
   client trains, Shamir-shares its secrets (SubmitSecureShares) and submits its
   masked update (SubmitMaskedUpdate), before the round deadline.
2. RECOVERY at the deadline the server fixes survivors (submitted) and dropped
   (did not), hands each responder the shares it holds plus the (owner, kind)
   list recovery needs (GetRevealShares), and collects the reveals
   (SubmitReveals). A second drop here is a responder that never answers; the
   Shamir threshold absorbs it.
3. COMBINE  the server cancels the masks and decodes the plain weighted mean.
"""

from __future__ import annotations

import logging
import time

import grpc
import numpy as np

from .config import Config
from .proto import fl_comm_pb2
from .secure_aggregation import (
    PUBLIC_KEY_BYTES,
    InsufficientSharesError,
    SecureAggregationError,
    Share,
)
from .secure_live import unflatten_weights
from .secure_round import RosterEntry, ServerSession
from .serialization import proto_nbytes, weights_to_proto
from .server import FederatedServer, RoundMetrics, _human_bytes

LOGGER = logging.getLogger("fl.secure_server")

_PB = fl_comm_pb2

# Wire width of a Shamir share value: a GF(2^521 - 1) element is at most 66 bytes.
_SHARE_VALUE_BYTES = 66


def _default_threshold(cohort_size: int) -> int:
    """A strict majority of the cohort. Recovery needs ``threshold`` shares to
    survive, so a majority tolerates any minority dropping — for a 5-client
    cohort that is 3, leaving room for up to two drops across both stages."""
    return cohort_size // 2 + 1


def _int_to_bytes(value: int, width: int) -> bytes:
    return int(value).to_bytes(width, "big")


class SecureFederatedServer(FederatedServer):
    """Federated server whose rounds run secure aggregation (no DP)."""

    def __init__(
        self,
        config: Config,
        initial_weights,
        evaluate_fn,
        *,
        threshold: int | None = None,
    ) -> None:
        # No DP aggregator: the secure round computes the weighted mean itself. A
        # trivial FedAvg passthrough satisfies the base contract but is never
        # called on the secure path (which overrides the whole round body).
        from .aggregation import FedAvgAggregator

        super().__init__(
            config=config,
            initial_weights=initial_weights,
            aggregator=FedAvgAggregator(),
            evaluate_fn=evaluate_fn,
            epsilon_fn=None,  # secure path is no-DP by construction
        )
        self._configured_threshold = threshold

        # Per-round secure state, all guarded by the inherited self._lock.
        self._sec_session: ServerSession | None = None
        self._sec_roster: dict[str, RosterEntry] = {}
        self._sec_threshold: int = 0
        self._sec_phase: str = "idle"  # idle | submit | recovery | closed
        self._sec_weights_msg = None
        self._sec_masked: set[str] = set()
        self._sec_reveal_needed: dict[str, list[tuple[str, str]]] = {}
        self._sec_held_shares: dict[str, list[Share]] = {}
        self._sec_reveals_in: set[str] = set()

    # -- roster -------------------------------------------------------------

    def _build_roster(self, cohort: list[str]) -> dict[str, RosterEntry]:
        """Assign each cohort member a contiguous order and pair it with the
        public key it announced at registration. Orders start at 0 in sorted-id
        order so both sides agree without an extra exchange."""
        with self._lock:
            records = {cid: self._clients[cid] for cid in cohort}
        roster: dict[str, RosterEntry] = {}
        for order, cid in enumerate(sorted(cohort)):
            public_key = records[cid].masking_public_key
            if not public_key:
                raise SecureAggregationError(
                    f"client {cid!r} registered without a masking public key; a secure "
                    "round needs every cohort member to be a V3 client"
                )
            roster[cid] = RosterEntry(order=order, public_key=public_key)
        return roster

    # -- the secure round ---------------------------------------------------

    def _run_one_round(self, round_index: int) -> RoundMetrics:  # noqa: PLR0915
        started = time.monotonic()
        cohort = self._sample_cohort()
        roster = self._build_roster(cohort)
        threshold = self._configured_threshold or _default_threshold(len(cohort))
        if not 1 <= threshold <= len(cohort):
            raise SecureAggregationError(
                f"threshold {threshold} is not in [1, cohort={len(cohort)}]"
            )

        session = ServerSession(threshold)
        for cid, entry in roster.items():
            session.announce(cid, entry.order, entry.public_key)
        session.roster_broadcast()  # log the broadcast bytes; roster is sent per poll

        weights_snapshot = self.global_weights()
        weights_msg = weights_to_proto(weights_snapshot, names=self._tensor_names)

        with self._lock:
            self._round = round_index
            self._cohort = set(cohort)
            self._sec_session = session
            self._sec_roster = roster
            self._sec_threshold = threshold
            self._sec_phase = "submit"
            self._sec_weights_msg = weights_msg
            self._sec_masked = set()
            self._sec_reveal_needed = {}
            self._sec_held_shares = {}
            self._sec_reveals_in = set()
            self._deadline_monotonic = started + self.config.server.round_deadline_seconds
            self._lock.notify_all()

        LOGGER.info(
            "secure round %d: cohort %s (threshold %d)",
            round_index,
            ", ".join(sorted(cohort)),
            threshold,
        )

        # Barrier 1: wait for every cohort member's masked update, or the deadline.
        with self._lock:
            while len(self._sec_masked) < len(cohort):
                remaining = self._deadline_monotonic - time.monotonic()
                if remaining <= 0:
                    break
                self._lock.wait(remaining)
            survivors = set(self._sec_masked)

        dropped = sorted(set(cohort) - survivors)
        for cid in dropped:
            LOGGER.warning(
                "secure round %d: %s did not submit before the deadline; recovering its "
                "pairwise masks from survivor shares",
                round_index,
                cid,
            )

        # Open recovery: fix the reveal requests and hand each responder its held
        # shares. A second drop during recovery is a responder that never answers.
        with self._lock:
            requests = session.reveal_requests(survivors)
            self._sec_reveal_needed = requests
            self._sec_held_shares = {rid: session.shares_for(rid) for rid in requests}
            self._sec_reveals_in = set()
            self._sec_phase = "recovery"
            self._deadline_monotonic = time.monotonic() + self.config.server.round_deadline_seconds
            self._lock.notify_all()

        # Barrier 2: wait for reveals from every responder, or the recovery deadline.
        with self._lock:
            while len(self._sec_reveals_in) < len(self._sec_reveal_needed):
                remaining = self._deadline_monotonic - time.monotonic()
                if remaining <= 0:
                    break
                self._lock.wait(remaining)
            self._deadline_monotonic = time.monotonic()
            self._sec_phase = "closed"

        aggregated = False
        recovered = None
        if not survivors:
            LOGGER.error("secure round %d: no survivors; keeping previous model", round_index)
        else:
            try:
                mean_flat, recovered = session.combine()
                shapes = [np.asarray(w).shape for w in weights_snapshot]
                new_weights = unflatten_weights(mean_flat, shapes)
                with self._lock:
                    self._global_weights = [np.asarray(w, dtype=np.float32) for w in new_weights]
                    self._model_version += 1
                aggregated = True
            except InsufficientSharesError:
                LOGGER.exception(
                    "secure round %d: recovery fell below threshold; keeping previous model",
                    round_index,
                )
            except SecureAggregationError:
                LOGGER.exception(
                    "secure round %d: recovery aborted; keeping previous model", round_index
                )

        loss, accuracy = self.evaluate_fn(self.global_weights())
        duration = time.monotonic() - started
        with self._lock:
            version = self._model_version

        secure_bytes = int(recovered["total_bytes"]) if recovered else session.total_bytes
        metrics = RoundMetrics(
            round=round_index,
            model_version=version,
            accuracy=accuracy,
            loss=loss,
            duration_seconds=duration,
            bytes_sent=proto_nbytes(weights_msg) * len(cohort),
            bytes_received=secure_bytes,
            num_selected=len(cohort),
            num_reported=len(survivors),
            num_dropped=len(dropped),
            dropped_clients=dropped,
            aggregated=aggregated,
            epsilon=None,
        )
        self.metrics.append(metrics)
        LOGGER.info(
            "secure round %d: acc=%.4f loss=%.4f %.2fs  reported=%d dropped=%d  secure_bytes=%s",
            round_index,
            accuracy,
            loss,
            duration,
            len(survivors),
            len(dropped),
            _human_bytes(secure_bytes),
        )
        return metrics

    # -- V3 RPC surface -----------------------------------------------------

    def GetSecureRound(self, request, context):  # noqa: N802
        with self._lock:
            if request.client_id not in self._clients:
                context.abort(grpc.StatusCode.NOT_FOUND, f"unknown client {request.client_id!r}")
            if self._finished:
                return _PB.GetSecureRoundResponse(
                    action=_PB.ROUND_ACTION_STOP,
                    round=self._round,
                    model_version=self._model_version,
                )
            sampled = request.client_id in self._cohort
            done = request.client_id in self._sec_masked
            if self._round == 0 or self._sec_phase != "submit" or not sampled or done:
                return _PB.GetSecureRoundResponse(
                    action=_PB.ROUND_ACTION_WAIT,
                    round=self._round,
                    model_version=self._model_version,
                )
            remaining = 0.0
            if self._deadline_monotonic is not None:
                remaining = max(0.0, self._deadline_monotonic - time.monotonic())
            t = self.config.training
            self._bytes_sent += proto_nbytes(self._sec_weights_msg)
            resp = _PB.GetSecureRoundResponse(
                action=_PB.ROUND_ACTION_TRAIN,
                round=self._round,
                model_version=self._model_version,
                weights=self._sec_weights_msg,
                seconds_until_deadline=remaining,
                local_epochs=t.local_epochs,
                batch_size=t.batch_size,
                learning_rate=t.learning_rate,
                momentum=t.momentum,
                threshold=self._sec_threshold,
            )
            for cid, entry in sorted(self._sec_roster.items()):
                resp.roster.add(
                    client_id=cid,
                    order=entry.order,
                    public_key=_int_to_bytes(entry.public_key, PUBLIC_KEY_BYTES),
                )
            return resp

    def SubmitSecureShares(self, request, context):  # noqa: N802
        with self._lock:
            reason = self._reject_reason(request.client_id, request.round)
            if reason is not None:
                return _PB.SubmitSecureSharesResponse(status=reason[0], detail=reason[1])
            parcels: dict[str, list[Share]] = {}
            for parcel in request.parcels:
                parcels[parcel.recipient_id] = [_share_from_proto(s) for s in parcel.shares]
            self._sec_session.route_shares(parcels)
            return _PB.SubmitSecureSharesResponse(status=_PB.UPDATE_STATUS_ACCEPTED)

    def SubmitMaskedUpdate(self, request, context):  # noqa: N802
        with self._lock:
            current_version = self._model_version
            reason = self._reject_reason(request.client_id, request.round)
            if reason is not None:
                return _PB.SubmitMaskedUpdateResponse(
                    status=reason[0], detail=reason[1], current_model_version=current_version
                )
            if request.model_version != current_version:
                return _PB.SubmitMaskedUpdateResponse(
                    status=_PB.UPDATE_STATUS_REJECTED_STALE_MODEL,
                    detail=(
                        f"masked update trained from v{request.model_version}, "
                        f"server holds v{current_version}"
                    ),
                    current_model_version=current_version,
                )
            words = np.frombuffer(request.masked_words, dtype="<u8")
            if words.size != request.num_words:
                return _PB.SubmitMaskedUpdateResponse(
                    status=_PB.UPDATE_STATUS_REJECTED_INVALID_PAYLOAD,
                    detail=(
                        f"masked_words carries {words.size} words, "
                        f"num_words says {request.num_words}"
                    ),
                    current_model_version=current_version,
                )
            self._sec_session.submit_masked(request.client_id, words.astype(np.uint64))
            self._sec_masked.add(request.client_id)
            self._bytes_received += len(request.masked_words)
            self._lock.notify_all()
            return _PB.SubmitMaskedUpdateResponse(
                status=_PB.UPDATE_STATUS_ACCEPTED, current_model_version=current_version
            )

    def GetRevealShares(self, request, context):  # noqa: N802
        with self._lock:
            not_ready = _PB.GetRevealResponse(ready=False, round=self._round)
            if self._finished or self._round != request.round:
                return not_ready
            if self._sec_phase != "recovery" or request.client_id not in self._sec_reveal_needed:
                # Not a responder, or recovery has not opened yet: poll again.
                return not_ready
            resp = _PB.GetRevealResponse(ready=True, round=self._round)
            for owner, kind in self._sec_reveal_needed[request.client_id]:
                resp.needed.add(owner=owner, kind=kind)
            for share in self._sec_held_shares.get(request.client_id, []):
                resp.held_shares.add(
                    owner=share.owner,
                    kind=share.kind,
                    index=int(share.index),
                    value=_int_to_bytes(int(share.value), _SHARE_VALUE_BYTES),
                )
            return resp

    def SubmitReveals(self, request, context):  # noqa: N802
        with self._lock:
            if self._round != request.round or self._sec_phase != "recovery":
                return _PB.SubmitRevealsResponse(
                    status=_PB.UPDATE_STATUS_REJECTED_DEADLINE_PASSED,
                    detail="recovery is not open for this round",
                )
            if request.client_id not in self._sec_reveal_needed:
                return _PB.SubmitRevealsResponse(
                    status=_PB.UPDATE_STATUS_REJECTED_NOT_SELECTED,
                    detail=f"{request.client_id!r} was not asked to reveal",
                )
            shares = [_share_from_proto(s) for s in request.shares]
            try:
                self._sec_session.accept_reveals(request.client_id, shares)
            except SecureAggregationError as exc:
                return _PB.SubmitRevealsResponse(
                    status=_PB.UPDATE_STATUS_REJECTED_INVALID_PAYLOAD, detail=str(exc)
                )
            self._sec_reveals_in.add(request.client_id)
            self._lock.notify_all()
            return _PB.SubmitRevealsResponse(status=_PB.UPDATE_STATUS_ACCEPTED)

    # -- helpers ------------------------------------------------------------

    def _reject_reason(self, client_id: str, round_index: int):
        """Common gate for the submit-phase RPCs. Returns ``(status, detail)`` to
        reject, or ``None`` to accept. Caller holds the lock."""
        if client_id not in self._clients:
            return _PB.UPDATE_STATUS_REJECTED_UNKNOWN_CLIENT, f"unknown client {client_id!r}"
        if self._round == 0 or self._finished or round_index != self._round:
            return (
                _PB.UPDATE_STATUS_REJECTED_DEADLINE_PASSED,
                f"round {round_index} is not open (server on {self._round})",
            )
        if client_id not in self._cohort:
            return (
                _PB.UPDATE_STATUS_REJECTED_NOT_SELECTED,
                f"{client_id!r} was not sampled for round {self._round}",
            )
        if self._sec_phase != "submit":
            return _PB.UPDATE_STATUS_REJECTED_DEADLINE_PASSED, "the submit phase has closed"
        if self._deadline_monotonic is not None and time.monotonic() > self._deadline_monotonic:
            return (
                _PB.UPDATE_STATUS_REJECTED_DEADLINE_PASSED,
                f"round {self._round} submit deadline has passed",
            )
        return None


def _share_from_proto(s) -> Share:
    return Share(
        owner=s.owner,
        kind=s.kind,
        index=int(s.index),
        value=int.from_bytes(s.value, "big"),
    )


def build_secure_server(config: Config, threshold: int | None = None) -> SecureFederatedServer:
    """Assemble a secure server from a config: model, held-out test set. No DP —
    the secure round computes the plain weighted mean and the DP path is
    deliberately excluded (see the module docstring)."""
    from .data import load_federated
    from .models import build_model
    from .server import build_evaluator

    if config.privacy.enabled:
        raise SecureAggregationError(
            "secure aggregation and the TFF DP path do not compose: DP clips and noises "
            "centrally, after seeing individual updates, which masking prevents. Run this "
            "path with privacy.enabled=false. See docs/architecture.md."
        )

    _train, test, _shards = load_federated(config.data, seed=config.seed)
    initial_weights = build_model(config.model.name, seed=config.seed).get_weights()
    LOGGER.info("secure server holds %d held-out test examples", len(test))
    return SecureFederatedServer(
        config=config,
        initial_weights=initial_weights,
        evaluate_fn=build_evaluator(config.model.name, test.x, test.y),
        threshold=threshold,
    )
