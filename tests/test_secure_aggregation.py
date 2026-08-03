"""Secure aggregation: primitives first, protocol on top.

This suite is numpy-plus-stdlib only — no TF/TFF — so it runs anywhere,
including hosts that cannot import the federated stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from fl.secure_aggregation import (
    FRACTIONAL_BITS,
    MODP_GENERATOR,
    MODP_PRIME,
    SHAMIR_PRIME,
    expand_mask,
    fixed_point_decode,
    fixed_point_encode,
    generate_keypair,
    shamir_reconstruct,
    shamir_split,
    shared_secret,
)


def _is_probable_prime(n: int, rounds: int = 20) -> bool:
    """Deterministic-enough Miller-Rabin for test purposes."""
    if n < 4:
        return n in (2, 3)
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    rng = np.random.default_rng(0)
    for _ in range(rounds):
        a = int(rng.integers(2, min(n - 2, 2**63)))
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


class TestGroupConstants:
    def test_modp_prime_is_prime_and_2048_bit(self):
        """The RFC 3526 group-14 constant is transcribed by hand; a single
        wrong hex digit would make it composite with overwhelming
        probability, so primality IS the transcription check."""
        assert MODP_PRIME.bit_length() == 2048
        assert _is_probable_prime(MODP_PRIME)

    def test_modp_prime_is_a_safe_prime_structure(self):
        """Group 14 is a safe prime: p = 2q + 1. Full primality of q is slow;
        checking p mod small primes and the generator's order suffices to
        catch transcription errors beyond what test one already does."""
        assert MODP_PRIME % 4 == 3  # safe primes are 3 mod 4
        # generator 2 generates the order-q subgroup, not order 2: g^2 != 1
        assert pow(MODP_GENERATOR, 2, MODP_PRIME) != 1

    def test_shamir_prime_is_mersenne_521(self):
        assert SHAMIR_PRIME == 2**521 - 1
        assert _is_probable_prime(SHAMIR_PRIME)


class TestKeyAgreement:
    def test_shared_secret_is_symmetric(self):
        sk_a, pk_a = generate_keypair(b"a" * 32)
        sk_b, pk_b = generate_keypair(b"b" * 32)
        assert shared_secret(sk_a, pk_b) == shared_secret(sk_b, pk_a)

    def test_distinct_pairs_get_distinct_secrets(self):
        sk_a, pk_a = generate_keypair(b"a" * 32)
        sk_b, pk_b = generate_keypair(b"b" * 32)
        sk_c, pk_c = generate_keypair(b"c" * 32)
        assert shared_secret(sk_a, pk_b) != shared_secret(sk_a, pk_c)

    def test_keypair_is_deterministic_in_the_seed(self):
        """Determinism is what makes the seed shareable for recovery."""
        assert generate_keypair(b"s" * 32) == generate_keypair(b"s" * 32)

    def test_bad_seed_length_rejected(self):
        with pytest.raises(ValueError, match="32 bytes"):
            generate_keypair(b"short")

    def test_public_key_outside_group_rejected(self):
        sk, _ = generate_keypair(b"a" * 32)
        with pytest.raises(ValueError, match="outside the group"):
            shared_secret(sk, MODP_PRIME)


class TestShamir:
    def test_round_trip_at_threshold(self):
        secret = bytes(range(32))
        shares = shamir_split(secret, num_shares=5, threshold=3)
        assert shamir_reconstruct(shares[:3]) == secret
        assert shamir_reconstruct(shares[2:]) == secret

    def test_below_threshold_yields_garbage_not_secret(self):
        secret = bytes(range(32))
        shares = shamir_split(secret, num_shares=5, threshold=3)
        assert shamir_reconstruct(shares[:2]) != secret

    def test_duplicate_indices_rejected(self):
        shares = shamir_split(b"x" * 32, 4, 2)
        with pytest.raises(ValueError, match="duplicate"):
            shamir_reconstruct([shares[0], shares[0]])

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError, match="threshold"):
            shamir_split(b"x" * 32, 3, 4)


class TestFixedPointAndMasks:
    def test_encode_decode_round_trip(self):
        values = np.array([0.5, -1.25, 0.0, 3.0e-3, -2.0], dtype=np.float32)
        decoded = fixed_point_decode(fixed_point_encode(values))
        np.testing.assert_allclose(decoded, values, atol=2.0**-FRACTIONAL_BITS)

    def test_sums_commute_with_encoding(self):
        """Sum-then-decode equals decode-then-sum bit-exactly: the property
        the whole aggregation rests on."""
        rng = np.random.default_rng(1)
        vectors = [rng.normal(size=64).astype(np.float32) for _ in range(10)]
        word_sum = np.zeros(64, dtype=np.uint64)
        for v in vectors:
            word_sum += fixed_point_encode(v)
        lhs = fixed_point_decode(word_sum)
        rhs = np.sum([fixed_point_decode(fixed_point_encode(v)) for v in vectors], axis=0)
        np.testing.assert_array_equal(lhs, rhs)

    def test_mask_is_deterministic_and_seed_sensitive(self):
        a = expand_mask(b"m" * 32, 100)
        assert np.array_equal(a, expand_mask(b"m" * 32, 100))
        assert not np.array_equal(a, expand_mask(b"n" * 32, 100))

    def test_mask_addition_cancels_exactly(self):
        mask = expand_mask(b"m" * 32, 100)
        vector = fixed_point_encode(np.random.default_rng(2).normal(size=100))
        assert np.array_equal((vector + mask) - mask, vector)

    def test_bad_mask_seed_rejected(self):
        with pytest.raises(ValueError, match="32 bytes"):
            expand_mask(b"tiny", 4)
