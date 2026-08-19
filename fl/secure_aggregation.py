"""Pairwise additive masking for secure aggregation — a TEACHING implementation.

                        *** NOT PRODUCTION CRYPTOGRAPHY ***

This module implements the core of Bonawitz et al. (CCS 2017), "Practical
Secure Aggregation for Privacy-Preserving Machine Learning", faithfully enough
to demonstrate and test the protocol's structure: pairwise masks that cancel
in the sum, a self-mask per client, Shamir-shared secrets so the sum survives
dropouts, and the either-or reveal rule that keeps a submitted update hidden
even when its owner later drops. It is written for one simulated process and
omits, deliberately and visibly, everything a deployment would demand:

- Share transport is SIMULATED. Shares move client-to-client through in-memory
  object references; in the real protocol they transit the server encrypted
  under pairwise keys. The server object here never reads them, but nothing
  cryptographic prevents it.
- Modular exponentiation uses Python ints: variable-time, cache-observable.
- No authentication anywhere: a real deployment needs identity binding on the
  key exchange (otherwise the server can sybil every pair) and signatures on
  round messages.
- No active-adversary defences: a malicious server that lies about who
  dropped can already break the honest-but-curious guarantees modelled here.

What the protocol DOES guarantee, and the tests assert: an honest-but-curious
server observes only uniformly-masked vectors and learns the aggregate alone;
the unmasked aggregate is bit-exact equal to the sum of the quantised inputs;
one client dropping mid-round, or a second client dropping during recovery,
degrades nothing so long as ``threshold`` shares survive.

Arithmetic note: updates are quantised to fixed point and masked in the group
Z_2^64 (uint64 wraparound), because float masks cannot cancel exactly.
"Exact" below always means bit-exact in the quantised domain; the float
round-trip error is the quantisation step, bounded by 2^-fractional_bits per
element per client.

Security note on what masking does NOT do: it constrains nothing about the
aggregate's contents. A malicious client's update — scaled, poisoned, or
NaN — aggregates exactly like any other, and the masking makes it *harder*
to attribute. This is the Byzantine limitation the README records; secure
aggregation widens it.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

import numpy as np

# RFC 3526 group 14: the 2048-bit MODP group, generator 2. A public, fixed,
# widely-deployed Diffie-Hellman group; its primality is asserted by the test
# suite (tests/test_secure_aggregation.py) so a transcription error here
# cannot survive CI.
MODP_PRIME = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74"
    "020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F1437"
    "4FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF05"
    "98DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB"
    "9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF695581718"
    "3995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)
MODP_GENERATOR = 2

# Shamir shares live in GF(p) for the Mersenne prime 2^521 - 1: comfortably
# larger than the 256-bit secrets being shared, and trivially correct to
# transcribe.
SHAMIR_PRIME = 2**521 - 1

FRACTIONAL_BITS = 24
SEED_BYTES = 32


class SecureAggregationError(RuntimeError):
    """Protocol violation or unrecoverable state."""


class InsufficientSharesError(SecureAggregationError):
    """Fewer than ``threshold`` shares survive for a secret the sum needs."""


# ---------------------------------------------------------------------------
# Key agreement
# ---------------------------------------------------------------------------


def generate_keypair(seed: bytes) -> tuple[int, int]:
    """Derive a Diffie-Hellman keypair from a 32-byte seed.

    The SEED is what gets Shamir-shared for dropout recovery — reconstructing
    it reconstructs the private key deterministically.
    """
    if len(seed) != SEED_BYTES:
        raise ValueError(f"seed must be {SEED_BYTES} bytes, got {len(seed)}")
    private = int.from_bytes(hashlib.sha512(b"secagg-dh-sk|" + seed).digest(), "big")
    private = private % (MODP_PRIME - 3) + 2  # in [2, p-2]
    public = pow(MODP_GENERATOR, private, MODP_PRIME)
    return private, public


def shared_secret(own_private: int, other_public: int) -> bytes:
    """The pairwise shared secret, hashed to 32 bytes. Symmetric by g^(ab)."""
    if not 2 <= other_public <= MODP_PRIME - 2:
        raise ValueError("public key outside the group")
    point = pow(other_public, own_private, MODP_PRIME)
    return hashlib.sha256(b"secagg-pair|" + point.to_bytes(256, "big")).digest()


# ---------------------------------------------------------------------------
# Shamir secret sharing over GF(2^521 - 1)
# ---------------------------------------------------------------------------


def shamir_split(secret: bytes, num_shares: int, threshold: int) -> list[tuple[int, int]]:
    """Split a secret of up to 64 bytes into ``num_shares`` points on a random
    degree-(threshold-1) polynomial; any ``threshold`` of them reconstruct."""
    value = int.from_bytes(secret, "big")
    if value >= SHAMIR_PRIME:
        raise ValueError("secret too large for the share field")
    if not 1 <= threshold <= num_shares:
        raise ValueError(f"need 1 <= threshold <= num_shares, got {threshold}/{num_shares}")
    coefficients = [value] + [secrets.randbelow(SHAMIR_PRIME) for _ in range(threshold - 1)]
    shares = []
    for x in range(1, num_shares + 1):
        y = 0
        for coefficient in reversed(coefficients):  # Horner
            y = (y * x + coefficient) % SHAMIR_PRIME
        shares.append((x, y))
    return shares


def shamir_reconstruct(shares: list[tuple[int, int]], secret_bytes: int = SEED_BYTES) -> bytes:
    """Lagrange interpolation at zero. Caller is responsible for supplying at
    least ``threshold`` distinct shares; fewer yields garbage, not an error —
    thresholds are enforced by the protocol layer, which knows them."""
    if len({x for x, _ in shares}) != len(shares):
        raise ValueError("duplicate share indices")
    total = 0
    for i, (xi, yi) in enumerate(shares):
        numerator, denominator = 1, 1
        for j, (xj, _) in enumerate(shares):
            if i == j:
                continue
            numerator = numerator * (-xj) % SHAMIR_PRIME
            denominator = denominator * (xi - xj) % SHAMIR_PRIME
        total = (total + yi * numerator * pow(denominator, -1, SHAMIR_PRIME)) % SHAMIR_PRIME
    # A below-threshold "reconstruction" is a near-uniform field element that
    # need not fit in secret_bytes; truncate so garbage stays garbage instead
    # of raising. A genuine reconstruction is < 2^(8*secret_bytes), unaffected.
    return (total % (1 << (8 * secret_bytes))).to_bytes(secret_bytes, "big")


# ---------------------------------------------------------------------------
# Fixed-point encoding and mask expansion
# ---------------------------------------------------------------------------


def fixed_point_encode(values: np.ndarray, fractional_bits: int = FRACTIONAL_BITS) -> np.ndarray:
    """Quantise floats into Z_2^64 as two's-complement fixed point."""
    scaled = np.rint(np.asarray(values, dtype=np.float64) * (1 << fractional_bits))
    return scaled.astype(np.int64).astype(np.uint64)


def fixed_point_decode(words: np.ndarray, fractional_bits: int = FRACTIONAL_BITS) -> np.ndarray:
    """Invert :func:`fixed_point_encode` (on sums as well as single vectors,
    provided the true sum stays inside int64 — the protocol layer's concern)."""
    return words.astype(np.int64).astype(np.float64) / (1 << fractional_bits)


def expand_mask(seed: bytes, length: int) -> np.ndarray:
    """Deterministically expand a 32-byte seed into ``length`` uint64 words."""
    if len(seed) != SEED_BYTES:
        raise ValueError(f"mask seed must be {SEED_BYTES} bytes, got {len(seed)}")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int.from_bytes(seed, "big"))))
    return rng.integers(0, 2**64, size=length, dtype=np.uint64)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

# Wire-size accounting, used by the message log and the cost model alike.
PUBLIC_KEY_BYTES = 256  # a 2048-bit group element
SHARE_BYTES = 68  # 66-byte field element plus 2-byte index
WORD_BYTES = 8  # masked vectors are uint64 per element

SELF_MASK = "self_mask"
KEY_SEED = "key_seed"


@dataclass(frozen=True)
class Share:
    """One Shamir share of one client secret, held by one other client."""

    owner: str
    kind: str  # SELF_MASK or KEY_SEED
    index: int
    value: int


class SecureClient:
    """One protocol participant. Holds its own secrets and everyone's shares.

    ``order`` is the client's position in the round's total ordering; it fixes
    the sign convention for pairwise masks (the lower order adds, the higher
    subtracts) so each pair's masks cancel in the sum.
    """

    def __init__(self, client_id: str, order: int, seed: bytes | None = None) -> None:
        self.client_id = client_id
        self.order = order
        root = seed if seed is not None else secrets.token_bytes(SEED_BYTES)
        self._key_seed = hashlib.sha256(b"secagg-key-seed|" + root).digest()
        self._self_mask_seed = hashlib.sha256(b"secagg-self-mask|" + root).digest()
        self._private, self.public_key = generate_keypair(self._key_seed)
        self._peers: dict[str, tuple[int, int]] = {}
        self._held_shares: dict[tuple[str, str], Share] = {}
        self._revealed: dict[str, str] = {}

    def receive_roster(self, roster: dict[str, tuple[int, int]]) -> None:
        """Learn every participant's (order, public key) from the server."""
        self._peers = {cid: entry for cid, entry in roster.items() if cid != self.client_id}

    def make_shares(
        self, roster: dict[str, tuple[int, int]], threshold: int
    ) -> dict[str, list[Share]]:
        """Shamir-share BOTH secrets among all participants (self included).

        Share x-coordinates are ``order + 1`` so every holder contributes a
        distinct point regardless of which secret is being reconstructed.
        """
        recipients = sorted(roster, key=lambda cid: roster[cid][0])
        n = len(recipients)
        out: dict[str, list[Share]] = {cid: [] for cid in recipients}
        for kind, secret in ((SELF_MASK, self._self_mask_seed), (KEY_SEED, self._key_seed)):
            points = shamir_split(secret, num_shares=n, threshold=threshold)
            for cid, (x, y) in zip(recipients, points, strict=True):
                if x != roster[cid][0] + 1:
                    raise SecureAggregationError("roster orders are not contiguous from zero")
                out[cid].append(Share(self.client_id, kind, x, y))
        return out

    def receive_shares(self, shares: list[Share]) -> None:
        for share in shares:
            self._held_shares[(share.owner, share.kind)] = share

    def _pair_mask_seed(self, other_id: str) -> bytes:
        _, other_public = self._peers[other_id]
        return hashlib.sha256(b"secagg-mask|" + shared_secret(self._private, other_public)).digest()

    def masked_update(self, values: np.ndarray, weight: float) -> np.ndarray:
        """The client's whole submission: fixed-point(update * weight, weight)
        plus the self mask plus every pairwise mask. The weight rides inside
        the masked vector, so the server does not even learn per-client
        example counts — only their sum."""
        if weight <= 0:
            raise ValueError("weight must be positive")
        payload = np.concatenate(
            [np.asarray(values, dtype=np.float64).ravel() * weight, [float(weight)]]
        )
        words = fixed_point_encode(payload)
        words = words + expand_mask(self._self_mask_seed, words.size)
        for other_id, (other_order, _) in self._peers.items():
            mask = expand_mask(self._pair_mask_seed(other_id), words.size)
            words = words + mask if self.order < other_order else words - mask
        return words

    def reveal(self, owner: str, kind: str) -> Share:
        """Reveal the held share of ``owner``'s ``kind`` secret.

        The protocol's crucial rule is enforced here: for any one owner a
        client reveals EITHER the self-mask seed (owner survived) OR the key
        seed (owner dropped), never both — the pair would let the server
        unmask that owner's individual submission.
        """
        already = self._revealed.get(owner)
        if already is not None and already != kind:
            raise SecureAggregationError(
                f"refusing to reveal both secrets of '{owner}': together they "
                f"would unmask an individual update"
            )
        share = self._held_shares.get((owner, kind))
        if share is None:
            raise SecureAggregationError(f"holding no {kind} share of '{owner}'")
        self._revealed[owner] = kind
        return share


class SecureServer:
    """The honest-but-curious aggregator: routes, sums, and unmasks the sum.

    Everything it observes goes through :attr:`message_log`, which is what the
    privacy tests interrogate: masked submissions, public keys, and revealed
    shares — never a plaintext update, never both secrets of one client.
    """

    def __init__(self, threshold: int) -> None:
        if threshold < 1:
            raise ValueError("threshold must be at least 1")
        self.threshold = threshold
        self.roster: dict[str, tuple[int, int]] = {}
        self.submissions: dict[str, np.ndarray] = {}
        self.message_log: list[dict] = []

    def _log(self, sender: str, receiver: str, kind: str, nbytes: int) -> None:
        self.message_log.append(
            {"sender": sender, "receiver": receiver, "kind": kind, "bytes": int(nbytes)}
        )

    def register(self, client: SecureClient) -> None:
        self.register_entry(client.client_id, client.order, client.public_key)

    def register_entry(self, client_id: str, order: int, public_key: int) -> None:
        """Register from primitives rather than a client object — what a
        distributed server does, holding only the (order, public key) a remote
        client announced, never the client itself."""
        self.roster[client_id] = (order, public_key)
        self._log(client_id, "server", "public_key", PUBLIC_KEY_BYTES)

    def broadcast_roster(self, clients: dict[str, SecureClient]) -> None:
        entry_bytes = PUBLIC_KEY_BYTES + 8
        for client in clients.values():
            client.receive_roster(self.roster)
            self._log("server", client.client_id, "roster", entry_bytes * len(self.roster))

    def route_shares(self, clients: dict[str, SecureClient]) -> None:
        """Move each client's shares to their holders. The honest server
        routes without reading; nothing but the docstring enforces that,
        which is one reason this is a teaching implementation."""
        for client in clients.values():
            per_recipient = client.make_shares(self.roster, self.threshold)
            for recipient_id, shares in per_recipient.items():
                self._log(client.client_id, recipient_id, "shares", SHARE_BYTES * len(shares))
                clients[recipient_id].receive_shares(shares)

    def submit(self, client_id: str, words: np.ndarray) -> None:
        if client_id not in self.roster:
            raise SecureAggregationError(f"'{client_id}' never registered")
        self.submissions[client_id] = words
        self._log(client_id, "server", "masked_update", words.size * WORD_BYTES)

    def reveals_needed(self) -> list[tuple[str, str]]:
        """The (owner, kind) shares recovery requires: every survivor's self-mask
        seed and every dropped client's key seed. This is the list a distributed
        server sends to each responder, and the same list :meth:`unmask` collects
        by calling into local client objects."""
        survivors = set(self.submissions)
        dropped = set(self.roster) - survivors
        return [(cid, SELF_MASK) for cid in sorted(survivors)] + [
            (cid, KEY_SEED) for cid in sorted(dropped)
        ]

    def unmask(
        self, clients: dict[str, SecureClient], responders: set[str]
    ) -> tuple[np.ndarray, dict]:
        """Recover the sum: subtract survivors' self masks, cancel dropped
        clients' pairwise masks, decode. ``responders`` are the survivors
        still answering during recovery — a second dropout at this stage is
        simply a survivor missing from this set.

        This is the in-process path: the server holds every client object and
        calls :meth:`SecureClient.reveal` directly. The distributed path collects
        the same shares as gRPC messages and hands them to :meth:`combine`; both
        share the arithmetic in :meth:`combine`."""
        needed = self.reveals_needed()
        survivors = set(self.submissions)
        collected: dict[tuple[str, str], list[tuple[int, int]]] = {key: [] for key in needed}
        for responder_id in sorted(responders & survivors):
            for owner, kind in needed:
                share = clients[responder_id].reveal(owner, kind)
                self._log(responder_id, "server", f"reveal_{kind}", SHARE_BYTES)
                collected[(owner, kind)].append((share.index, share.value))
        return self.combine(collected, responders=responders & survivors)

    def combine(
        self,
        collected: dict[tuple[str, str], list[tuple[int, int]]],
        responders: set[str] | frozenset[str] = frozenset(),
    ) -> tuple[np.ndarray, dict]:
        """Turn the masked submissions plus the revealed shares into the plain
        aggregate. Pure arithmetic over data — no client objects — so the gRPC
        server, which never holds a remote client, calls exactly this after
        gathering reveals as messages.

        ``collected`` maps each needed ``(owner, kind)`` to the list of
        ``(share_index, share_value)`` pairs that responders revealed for it.
        """
        survivors = set(self.submissions)
        dropped = set(self.roster) - survivors
        for (owner, kind), shares in collected.items():
            if len(shares) < self.threshold:
                raise InsufficientSharesError(
                    f"{len(shares)} shares of {owner}/{kind} survive; threshold is {self.threshold}"
                )

        sizes = {words.size for words in self.submissions.values()}
        if len(sizes) != 1:
            raise SecureAggregationError(f"submission lengths disagree: {sorted(sizes)}")
        (length,) = sizes
        total = np.zeros(length, dtype=np.uint64)
        for words in self.submissions.values():
            total += words

        for cid in survivors:
            seed = shamir_reconstruct(collected[(cid, SELF_MASK)][: self.threshold])
            total -= expand_mask(seed, length)

        for cid in dropped:
            key_seed = shamir_reconstruct(collected[(cid, KEY_SEED)][: self.threshold])
            private, public = generate_keypair(key_seed)
            if public != self.roster[cid][1]:
                raise SecureAggregationError(
                    f"reconstructed key for '{cid}' does not match its roster entry"
                )
            dropped_order = self.roster[cid][0]
            for survivor_id in survivors:
                survivor_order, survivor_public = self.roster[survivor_id]
                pair_secret = shared_secret(private, survivor_public)
                mask = expand_mask(hashlib.sha256(b"secagg-mask|" + pair_secret).digest(), length)
                # Cancel the term the SURVIVOR added for this pair.
                total = total - mask if survivor_order < dropped_order else total + mask

        payload = fixed_point_decode(total)
        weight_sum = payload[-1]
        if weight_sum <= 0:
            raise SecureAggregationError("aggregate weight is not positive")
        report = {
            "survivors": sorted(survivors),
            "dropped": sorted(dropped),
            "responders": sorted(set(responders) & survivors),
            "weight_sum": float(weight_sum),
            "total_bytes": int(sum(m["bytes"] for m in self.message_log)),
        }
        return payload[:-1] / weight_sum, report


def run_secure_round(
    updates: list[tuple[str, np.ndarray, float]],
    threshold: int,
    drop_before_submit: set[str] | frozenset[str] = frozenset(),
    drop_during_recovery: set[str] | frozenset[str] = frozenset(),
) -> tuple[np.ndarray, dict]:
    """One full round over in-memory clients: setup, masking, dropouts,
    recovery. Returns the weighted mean of the SURVIVORS' updates and the
    server's report. Clients in ``drop_before_submit`` complete setup but
    never submit; clients in ``drop_during_recovery`` submit but go silent
    before revealing any shares."""
    ids = [cid for cid, _, _ in updates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate client ids")
    unknown = (set(drop_before_submit) | set(drop_during_recovery)) - set(ids)
    if unknown:
        raise ValueError(f"drop sets name unknown clients: {sorted(unknown)}")
    if set(drop_before_submit) & set(drop_during_recovery):
        raise ValueError("a client cannot drop both before submission and during recovery")

    clients = {cid: SecureClient(cid, order) for order, cid in enumerate(ids)}
    server = SecureServer(threshold)
    for client in clients.values():
        server.register(client)
    server.broadcast_roster(clients)
    server.route_shares(clients)

    for cid, values, weight in updates:
        if cid in drop_before_submit:
            continue
        server.submit(cid, clients[cid].masked_update(values, weight))

    responders = set(ids) - set(drop_before_submit) - set(drop_during_recovery)
    return server.unmask(clients, responders)


# ---------------------------------------------------------------------------
# What masking costs
# ---------------------------------------------------------------------------


def communication_cost(
    num_clients: int,
    num_params: int,
    dropouts: int = 0,
    recovery_silent: int = 0,
) -> dict:
    """Bytes moved in one secure round versus plain FedAvg, by the same
    accounting the server's message log uses (the tests hold the two equal).

    Counts each logged message once. Client-to-client shares are routed
    through the server, so their wire cost is really two hops; the model
    single-counts them, which UNDERSTATES the true share traffic by 2x —
    stated here rather than hidden. Plain FedAvg is modelled as one float32
    update upload per client; its example-count integer is noise.
    """
    if dropouts + recovery_silent >= num_clients:
        raise ValueError("at least one client must survive to respond")
    n = num_clients
    submitted = n - dropouts
    responders = submitted - recovery_silent
    vector_bytes = (num_params + 1) * WORD_BYTES
    breakdown = {
        "registration": n * PUBLIC_KEY_BYTES,
        "roster_broadcast": n * n * (PUBLIC_KEY_BYTES + 8),
        "share_distribution": n * n * 2 * SHARE_BYTES,
        "masked_updates": submitted * vector_bytes,
        "recovery_reveals": responders * n * SHARE_BYTES,
    }
    secure_total = sum(breakdown.values())
    plain_total = n * num_params * 4
    return {
        "num_clients": n,
        "num_params": num_params,
        "dropouts": dropouts,
        "recovery_silent": recovery_silent,
        "breakdown": breakdown,
        "secure_total_bytes": secure_total,
        "plain_total_bytes": plain_total,
        "overhead_ratio": secure_total / plain_total,
        "per_client_upload_secure_bytes": vector_bytes,
        "per_client_upload_plain_bytes": num_params * 4,
    }
