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
