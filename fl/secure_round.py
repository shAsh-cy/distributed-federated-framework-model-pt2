"""Message-driven secure aggregation — the transport-agnostic orchestration.

:mod:`fl.secure_aggregation` runs a secure round in ONE process, the server
holding every client object and calling their methods directly
(``run_secure_round``). The live path cannot: clients are separate processes and
the server reaches them only by message. This module re-expresses the same round
as an exchange of messages between a :class:`ServerSession` and one
:class:`ParticipantSession` per client, holding no cross-process references.

Everything here is numpy + stdlib, so the full orchestration — setup, share
routing, masked submission, dropout, recovery, the either-or reveal rule — is
driven and tested in-process by :func:`run_round`, WITHOUT gRPC. The gRPC layer
(:mod:`fl.secure_server`, :mod:`fl.secure_client`) is then a thin transport that
carries exactly these messages; its only job is faithful delivery, and the
arithmetic it relies on is the tested code below and in
:meth:`fl.secure_aggregation.SecureServer.combine`.

The phases, in order, and which side drives each:

1. ANNOUNCE   client -> server: (order, public key). Keys are derived at
   registration, so this is the client's first round message.
2. ROSTER     server -> client: every cohort member's (order, public key) and
   the threshold. A client needs only this to mask.
3. SHARES     client -> server -> client: each client Shamir-shares both its
   secrets and the server routes one share to each holder. Shares are held by
   CLIENTS, never the server — that is what stops the server reconstructing an
   individual, and the client enforces the either-or reveal on top.
4. MASK       client -> server: the masked update words.
5. REVEAL     server -> client -> server: after the deadline the server asks each
   responder for the shares recovery needs (survivors' self-mask seeds, dropped
   clients' key seeds) and combines them into the plain aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .secure_aggregation import (
    PUBLIC_KEY_BYTES,
    SHARE_BYTES,
    SecureAggregationError,
    SecureClient,
    SecureServer,
    Share,
)


@dataclass(frozen=True)
class RosterEntry:
    """One cohort member as everyone else sees it: its ordering position and
    public key. No secret, so this is safe to broadcast to the whole cohort."""

    order: int
    public_key: int


class ParticipantSession:
    """One client's side of a secure round, wrapping a :class:`SecureClient`.

    Every method is pure message-in / message-out: it takes what the transport
    delivered and returns what the transport must send next, holding no
    reference to the server or to any peer object.
    """

    def __init__(self, client_id: str, order: int, seed: bytes | None = None) -> None:
        self._client = SecureClient(client_id, order, seed=seed)
        self.client_id = client_id
        self.order = order
        self._threshold: int | None = None

    @property
    def public_key(self) -> int:
        return self._client.public_key

    def make_share_parcels(
        self, roster: dict[str, RosterEntry], threshold: int
    ) -> dict[str, list[Share]]:
        """Learn the cohort, then Shamir-share both secrets, one share per member.
        Returns ``{recipient_id: [shares]}`` for the server to route."""
        self._threshold = threshold
        as_tuples = {cid: (e.order, e.public_key) for cid, e in roster.items()}
        self._client.receive_roster(as_tuples)
        return self._client.make_shares(as_tuples, threshold)

    def receive_shares(self, shares: list[Share]) -> None:
        self._client.receive_shares(shares)

    def masked_update(self, values: np.ndarray, weight: float) -> np.ndarray:
        return self._client.masked_update(values, weight)

    def reveal(self, needed: list[tuple[str, str]]) -> list[Share]:
        """Reveal exactly the shares recovery asked for. The either-or rule is
        enforced inside :meth:`SecureClient.reveal`: a request for both secrets
        of one owner raises rather than handing the server an unmasking pair."""
        return [self._client.reveal(owner, kind) for owner, kind in needed]


@dataclass
class ServerSession:
    """The server's side of a secure round: routes messages, sums masked words,
    and combines the reveals into the plain aggregate. Holds no client objects.

    Backed by a :class:`SecureServer` for the roster, the masked-word sum and the
    reveal arithmetic (:meth:`SecureServer.combine`); this class adds only the
    server-side buffering the distributed case needs — routed shares and gathered
    reveals — which the in-process ``SecureServer`` did by reaching into client
    objects.
    """

    threshold: int
    _server: SecureServer = field(init=False)
    _share_buffer: dict[str, list[Share]] = field(init=False, default_factory=dict)
    _collected: dict[tuple[str, str], list[tuple[int, int]]] = field(
        init=False, default_factory=dict
    )
    _responders: set[str] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self._server = SecureServer(self.threshold)

    # -- setup --------------------------------------------------------------

    def announce(self, client_id: str, order: int, public_key: int) -> None:
        self._server.register_entry(client_id, order, public_key)

    @property
    def roster(self) -> dict[str, RosterEntry]:
        return {cid: RosterEntry(order, pk) for cid, (order, pk) in self._server.roster.items()}

    def roster_broadcast(self) -> dict[str, RosterEntry]:
        """The roster to send to every cohort member, logging one broadcast per
        member so the byte accounting matches the in-process ``broadcast_roster``
        (each entry costs a public key plus an 8-byte order)."""
        entry_bytes = PUBLIC_KEY_BYTES + 8
        roster = self.roster
        for client_id in self._server.roster:
            self._server._log("server", client_id, "roster", entry_bytes * len(roster))
        return roster

    def route_shares(self, parcels: dict[str, list[Share]]) -> None:
        """Buffer one client's outgoing shares by recipient. Wire cost is logged
        the same way :meth:`SecureServer.route_shares` logs it, so the analytic
        communication_cost model still reconciles to the byte."""
        for recipient_id, shares in parcels.items():
            if not shares:
                continue
            sender = shares[0].owner
            self._server._log(sender, recipient_id, "shares", SHARE_BYTES * len(shares))
            self._share_buffer.setdefault(recipient_id, []).extend(shares)

    def shares_for(self, client_id: str) -> list[Share]:
        """The shares buffered for one client, to deliver to it."""
        return list(self._share_buffer.get(client_id, []))

    # -- submission and recovery -------------------------------------------

    def submit_masked(self, client_id: str, words: np.ndarray) -> None:
        self._server.submit(client_id, words)

    def reveal_requests(self, responders: set[str]) -> dict[str, list[tuple[str, str]]]:
        """After the deadline: which (owner, kind) shares each responder must
        reveal. Survivors are whoever submitted masked words; the dropped are the
        rest. ``responders`` are the survivors still answering — a second dropout
        here is just a survivor absent from this set."""
        survivors = set(self._server.submissions)
        self._responders = set(responders) & survivors
        needed = self._server.reveals_needed()
        self._collected = {key: [] for key in needed}
        return {rid: list(needed) for rid in sorted(self._responders)}

    def accept_reveals(self, responder_id: str, shares: list[Share]) -> None:
        if responder_id not in self._responders:
            raise SecureAggregationError(f"'{responder_id}' was not asked to reveal")
        for share in shares:
            key = (share.owner, share.kind)
            if key not in self._collected:
                raise SecureAggregationError(f"unexpected reveal of {key} from '{responder_id}'")
            self._server._log(responder_id, "server", f"reveal_{share.kind}", SHARE_BYTES)
            self._collected[key].append((share.index, share.value))

    def combine(self) -> tuple[np.ndarray, dict]:
        """The plain aggregate from the masked sum plus the gathered reveals."""
        return self._server.combine(self._collected, responders=self._responders)

    @property
    def total_bytes(self) -> int:
        return int(sum(m["bytes"] for m in self._server.message_log))


def run_round(
    updates: list[tuple[str, np.ndarray, float]],
    threshold: int,
    drop_before_submit: set[str] | frozenset[str] = frozenset(),
    drop_during_recovery: set[str] | frozenset[str] = frozenset(),
) -> tuple[np.ndarray, dict]:
    """Drive a whole secure round through the message API, in-process.

    This is the distributed orchestration with an in-memory transport: it makes
    exactly the message exchanges the gRPC layer makes, so a green result here is
    evidence the *sequencing* is right independently of the wire. It returns the
    same weighted mean and report as :func:`fl.secure_aggregation.run_secure_round`
    — and a test holds the two equal.
    """
    ids = [cid for cid, _, _ in updates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate client ids")

    participants = {
        cid: ParticipantSession(cid, order) for order, (cid, _, _) in enumerate(updates)
    }
    server = ServerSession(threshold)

    # 1-2. Announce, then the server broadcasts the roster to every member.
    for p in participants.values():
        server.announce(p.client_id, p.order, p.public_key)
    roster = server.roster_broadcast()

    # 3. Each client shares its secrets; the server routes; each client receives.
    for p in participants.values():
        server.route_shares(p.make_share_parcels(roster, threshold))
    for p in participants.values():
        p.receive_shares(server.shares_for(p.client_id))

    # 4. Masked submission, minus the clients that drop before submitting.
    for cid, values, weight in updates:
        if cid in drop_before_submit:
            continue
        server.submit_masked(cid, participants[cid].masked_update(values, weight))

    # 5. Recovery: ask responders to reveal, gather, combine.
    responders = set(ids) - set(drop_before_submit) - set(drop_during_recovery)
    requests = server.reveal_requests(responders)
    for rid, needed in requests.items():
        server.accept_reveals(rid, participants[rid].reveal(needed))
    return server.combine()
