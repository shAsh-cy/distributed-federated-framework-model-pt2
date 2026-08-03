"""Secure aggregation: primitives first, protocol on top.

This suite is numpy-plus-stdlib only — no TF/TFF — so it runs anywhere,
including hosts that cannot import the federated stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from fl.secure_aggregation import (
    FRACTIONAL_BITS,
    KEY_SEED,
    MODP_GENERATOR,
    MODP_PRIME,
    SELF_MASK,
    SHAMIR_PRIME,
    InsufficientSharesError,
    SecureAggregationError,
    SecureClient,
    SecureServer,
    expand_mask,
    fixed_point_decode,
    fixed_point_encode,
    generate_keypair,
    run_secure_round,
    shamir_reconstruct,
    shamir_split,
    shared_secret,
)


def _updates(n: int, size: int = 32, seed: int = 7) -> list[tuple[str, np.ndarray, float]]:
    rng = np.random.default_rng(seed)
    return [
        (f"c{i}", rng.normal(scale=0.5, size=size).astype(np.float32), float(10 + 7 * i))
        for i in range(n)
    ]


def _quantised_weighted_mean(updates: list[tuple[str, np.ndarray, float]]) -> np.ndarray:
    """The same arithmetic with no masks anywhere: the ground truth the
    protocol must reproduce BIT-EXACTLY."""
    length = updates[0][1].size + 1
    total = np.zeros(length, dtype=np.uint64)
    for _, values, weight in updates:
        payload = np.concatenate([values.astype(np.float64).ravel() * weight, [float(weight)]])
        total += fixed_point_encode(payload)
    decoded = fixed_point_decode(total)
    return decoded[:-1] / decoded[-1]


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


class TestFullRound:
    def test_unmasked_sum_equals_plain_sum_exactly(self):
        """The headline property: with every mask applied and removed, the
        aggregate is bit-identical to the maskless quantised computation, and
        within quantisation error of the float weighted mean."""
        updates = _updates(5)
        result, report = run_secure_round(updates, threshold=3)
        np.testing.assert_array_equal(result, _quantised_weighted_mean(updates))
        weights = np.array([w for _, _, w in updates])
        stacked = np.stack([v.astype(np.float64) for _, v, _ in updates])
        float_mean = (weights[:, None] * stacked).sum(axis=0) / weights.sum()
        np.testing.assert_allclose(result, float_mean, atol=1e-5)
        assert report["dropped"] == []
        assert report["weight_sum"] == pytest.approx(weights.sum())

    def test_server_never_observes_a_plaintext_update(self):
        """What the server holds is uniformly masked: no submission equals the
        plaintext encoding, and even with the self mask stripped (which the
        server CAN do after reveals) the pairwise masks still cover it."""
        updates = _updates(4)
        clients = {cid: SecureClient(cid, order) for order, (cid, _, _) in enumerate(updates)}
        server = SecureServer(threshold=3)
        for client in clients.values():
            server.register(client)
        server.broadcast_roster(clients)
        server.route_shares(clients)
        for cid, values, weight in updates:
            server.submit(cid, clients[cid].masked_update(values, weight))
        for cid, values, weight in updates:
            payload = np.concatenate(
                [values.astype(np.float64).ravel() * weight, [float(weight)]]
            )
            plaintext = fixed_point_encode(payload)
            observed = server.submissions[cid]
            assert not np.array_equal(observed, plaintext)
            self_stripped = observed - expand_mask(clients[cid]._self_mask_seed, observed.size)
            assert not np.array_equal(self_stripped, plaintext)

    def test_mid_round_dropout_recovers_the_survivor_sum_exactly(self):
        updates = _updates(5)
        result, report = run_secure_round(updates, threshold=3, drop_before_submit={"c2"})
        survivors = [u for u in updates if u[0] != "c2"]
        np.testing.assert_array_equal(result, _quantised_weighted_mean(survivors))
        assert report["dropped"] == ["c2"]

    def test_second_dropout_during_recovery_still_recovers(self):
        """c0 vanishes before submitting; c1 submits, then vanishes during
        recovery. Four responders remain against a threshold of three, so the
        sum over the five SUBMITTED clients is still exact."""
        updates = _updates(6)
        result, report = run_secure_round(
            updates,
            threshold=3,
            drop_before_submit={"c0"},
            drop_during_recovery={"c1"},
        )
        submitted = [u for u in updates if u[0] != "c0"]
        np.testing.assert_array_equal(result, _quantised_weighted_mean(submitted))
        assert report["dropped"] == ["c0"]
        assert "c1" in report["survivors"]
        assert "c1" not in report["responders"]

    def test_dropout_below_threshold_is_an_explicit_abort(self):
        updates = _updates(4)
        with pytest.raises(InsufficientSharesError, match="threshold"):
            run_secure_round(updates, threshold=4, drop_during_recovery={"c3"})

    def test_client_refuses_to_reveal_both_secrets(self):
        """The either-or rule that keeps a submitted-then-dropped client's
        update hidden: one holder will never hand the server both the self
        mask and the key seed of the same owner."""
        updates = _updates(3)
        clients = {cid: SecureClient(cid, order) for order, (cid, _, _) in enumerate(updates)}
        server = SecureServer(threshold=2)
        for client in clients.values():
            server.register(client)
        server.broadcast_roster(clients)
        server.route_shares(clients)
        holder = clients["c0"]
        holder.reveal("c1", SELF_MASK)
        with pytest.raises(SecureAggregationError, match="both secrets"):
            holder.reveal("c1", KEY_SEED)

    def test_reconstructed_key_must_match_the_roster(self):
        """Integrity check on recovery: if the shares reconstruct a key that
        does not match the dropped client's registered public key, the round
        aborts instead of subtracting garbage masks."""
        updates = _updates(4)
        clients = {cid: SecureClient(cid, order) for order, (cid, _, _) in enumerate(updates)}
        server = SecureServer(threshold=2)
        for client in clients.values():
            server.register(client)
        server.broadcast_roster(clients)
        server.route_shares(clients)
        for cid, values, weight in updates:
            if cid == "c1":
                continue
            server.submit(cid, clients[cid].masked_update(values, weight))
        order, _ = server.roster["c1"]
        server.roster["c1"] = (order, MODP_GENERATOR)  # tampered registration
        with pytest.raises(SecureAggregationError, match="roster"):
            server.unmask(clients, {"c0", "c2", "c3"})

    def test_driver_input_validation(self):
        updates = _updates(3)
        with pytest.raises(ValueError, match="duplicate"):
            run_secure_round([updates[0], updates[0], updates[2]], threshold=2)
        with pytest.raises(ValueError, match="unknown"):
            run_secure_round(updates, threshold=2, drop_before_submit={"ghost"})
        with pytest.raises(ValueError, match="cannot drop both"):
            run_secure_round(
                updates, threshold=2, drop_before_submit={"c0"}, drop_during_recovery={"c0"}
            )
        with pytest.raises(ValueError, match="at least 1"):
            SecureServer(threshold=0)
        with pytest.raises(SecureAggregationError, match="never registered"):
            SecureServer(threshold=1).submit("c9", np.zeros(3, dtype=np.uint64))
        with pytest.raises(ValueError, match="positive"):
            SecureClient("c0", 0).masked_update(np.zeros(3, dtype=np.float32), 0.0)
