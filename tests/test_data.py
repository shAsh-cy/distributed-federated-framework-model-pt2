"""Tests for dataset loading and client partitioning.

The three invariants demanded of any partition -- disjointness, exhaustiveness,
and no test-set leakage -- are checked for both schemes and across several seeds,
because a partitioner that only happens to be correct for one seed is not correct.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from fl.data import (
    NUM_CLASSES,
    Dataset,
    label_distribution,
    load_fashion_mnist,
    partition,
    partition_dirichlet,
    partition_iid,
    partition_summary,
)

SEEDS = (0, 1, 42)
SCHEMES = ("iid", "dirichlet")


@pytest.fixture(scope="module")
def fashion():
    """Fashion-MNIST, loaded once for the whole module."""
    return load_fashion_mnist()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def test_shapes_and_ranges(fashion):
    train, test = fashion
    assert train.x.shape == (60_000, 28, 28, 1)
    assert test.x.shape == (10_000, 28, 28, 1)
    assert train.x.dtype == np.float32
    assert 0.0 <= train.x.min() and train.x.max() <= 1.0
    assert set(np.unique(train.y)) == set(range(NUM_CLASSES))


def test_train_and_test_are_returned_separately(fashion):
    """The loader hands back two objects; nothing merges them."""
    train, test = fashion
    assert isinstance(train, Dataset) and isinstance(test, Dataset)
    assert len(train) == 60_000
    assert len(test) == 10_000


def test_take_returns_a_copy(fashion):
    train, _ = fashion
    sub = train.take(np.array([0, 1, 2]))
    sub.x[0, 0, 0, 0] = 99.0
    assert train.x[0, 0, 0, 0] != 99.0


# ---------------------------------------------------------------------------
# Invariant 1: shards are disjoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("seed", SEEDS)
def test_client_shards_are_pairwise_disjoint(fashion, scheme, seed):
    train, _ = fashion
    shards = partition(train.y, num_clients=10, scheme=scheme, alpha=0.5, seed=seed)
    for i in range(len(shards)):
        for j in range(i + 1, len(shards)):
            overlap = np.intersect1d(shards[i], shards[j])
            assert overlap.size == 0, f"clients {i} and {j} share {overlap.size} samples"


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("seed", SEEDS)
def test_no_sample_is_duplicated_anywhere(fashion, scheme, seed):
    train, _ = fashion
    shards = partition(train.y, num_clients=10, scheme=scheme, alpha=0.5, seed=seed)
    everything = np.concatenate(shards)
    assert everything.size == np.unique(everything).size


# ---------------------------------------------------------------------------
# Invariant 2: the union is the whole training set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("seed", SEEDS)
def test_union_of_shards_is_the_entire_train_set(fashion, scheme, seed):
    train, _ = fashion
    shards = partition(train.y, num_clients=10, scheme=scheme, alpha=0.5, seed=seed)
    union = np.unique(np.concatenate(shards))
    np.testing.assert_array_equal(union, np.arange(len(train)))
    assert sum(s.size for s in shards) == len(train) == 60_000


@pytest.mark.parametrize("num_clients", [1, 2, 5, 10, 37])
def test_exhaustive_for_various_client_counts(fashion, num_clients):
    train, _ = fashion
    for scheme in SCHEMES:
        shards = partition(train.y, num_clients, scheme=scheme, alpha=0.5, seed=7)
        assert len(shards) == num_clients
        assert sum(s.size for s in shards) == len(train)
        assert np.unique(np.concatenate(shards)).size == len(train)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_no_client_receives_an_empty_shard(fashion, scheme):
    train, _ = fashion
    for seed in SEEDS:
        shards = partition(train.y, num_clients=20, scheme=scheme, alpha=0.5, seed=seed)
        assert all(s.size > 0 for s in shards)


def test_extreme_alpha_still_yields_non_empty_shards(fashion):
    """At alpha=0.01 classes collapse onto single clients; repair must still hold."""
    train, _ = fashion
    shards = partition(train.y, num_clients=20, scheme="dirichlet", alpha=0.01, seed=3)
    assert all(s.size > 0 for s in shards)
    assert sum(s.size for s in shards) == len(train)
    assert np.unique(np.concatenate(shards)).size == len(train)


# ---------------------------------------------------------------------------
# Invariant 3: no test sample reaches any client
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
def test_no_test_sample_appears_in_any_client_shard(fashion, scheme):
    """Content-level check, not an index-level one.

    Comparing indices would only prove the partitioner indexes into the training
    array. Hashing the actual pixels proves no held-out image reached a client
    even if the two arrays were ever accidentally concatenated upstream.
    """
    train, test = fashion
    shards = partition(train.y, num_clients=10, scheme=scheme, alpha=0.5, seed=42)

    test_hashes = {hashlib.md5(row.tobytes()).hexdigest() for row in test.x}
    assert len(test_hashes) == len(test), "precondition: test images are unique"

    for cid, shard in enumerate(shards):
        client_data = train.take(shard)
        client_hashes = {hashlib.md5(row.tobytes()).hexdigest() for row in client_data.x}
        leaked = client_hashes & test_hashes
        assert not leaked, f"client {cid} holds {len(leaked)} held-out test images"


def test_partition_never_sees_the_test_set(fashion):
    """Every index a client receives must be a valid training index."""
    train, test = fashion
    shards = partition(train.y, num_clients=10, scheme="dirichlet", alpha=0.5, seed=42)
    for shard in shards:
        assert shard.min() >= 0
        assert shard.max() < len(train)
    assert sum(s.size for s in shards) == len(train)
    assert len(test) == 10_000  # untouched


# ---------------------------------------------------------------------------
# The schemes must actually differ
# ---------------------------------------------------------------------------


def _max_total_variation(labels: np.ndarray, shards: list[np.ndarray]) -> float:
    """Largest TV distance between any client's label distribution and the global one."""
    global_dist = np.bincount(labels, minlength=NUM_CLASSES) / len(labels)
    worst = 0.0
    for shard in shards:
        local = label_distribution(labels, shard)
        local = local / local.sum()
        worst = max(worst, 0.5 * np.abs(local - global_dist).sum())
    return worst


def test_iid_split_is_close_to_the_global_label_distribution(fashion):
    train, _ = fashion
    shards = partition(train.y, num_clients=10, scheme="iid", seed=42)
    assert _max_total_variation(train.y, shards) < 0.05


def test_iid_shards_are_equal_sized(fashion):
    train, _ = fashion
    sizes = [s.size for s in partition(train.y, num_clients=10, scheme="iid", seed=42)]
    assert max(sizes) - min(sizes) <= 1


def test_dirichlet_split_is_genuinely_skewed(fashion):
    """The default must not be quietly IID -- that is the whole point of the split."""
    train, _ = fashion
    shards = partition(train.y, num_clients=10, scheme="dirichlet", alpha=0.5, seed=42)
    assert _max_total_variation(train.y, shards) > 0.25


def test_lower_alpha_produces_more_skew(fashion):
    """Monotonicity in alpha is the property that makes it a meaningful knob."""
    train, _ = fashion
    skews = [
        _max_total_variation(train.y, partition(train.y, 10, scheme="dirichlet", alpha=a, seed=42))
        for a in (100.0, 1.0, 0.1)
    ]
    assert skews[0] < skews[1] < skews[2]


def test_dirichlet_shard_sizes_are_unequal(fashion):
    """Unequal shards are what make sample-count weighting in FedAvg matter."""
    train, _ = fashion
    sizes = [s.size for s in partition(train.y, 10, scheme="dirichlet", alpha=0.5, seed=42)]
    assert max(sizes) > 2 * min(sizes)


# ---------------------------------------------------------------------------
# Determinism and error handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
def test_partition_is_deterministic_given_a_seed(fashion, scheme):
    train, _ = fashion
    a = partition(train.y, 10, scheme=scheme, alpha=0.5, seed=99)
    b = partition(train.y, 10, scheme=scheme, alpha=0.5, seed=99)
    for x, y in zip(a, b, strict=False):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_different_seeds_give_different_partitions(fashion, scheme):
    train, _ = fashion
    a = partition(train.y, 10, scheme=scheme, alpha=0.5, seed=1)
    b = partition(train.y, 10, scheme=scheme, alpha=0.5, seed=2)
    assert any(not np.array_equal(x, y) for x, y in zip(a, b, strict=False))


def test_shards_are_sorted(fashion):
    train, _ = fashion
    for shard in partition(train.y, 10, scheme="dirichlet", alpha=0.5, seed=0):
        np.testing.assert_array_equal(shard, np.sort(shard))


def test_unknown_scheme_rejected(fashion):
    train, _ = fashion
    with pytest.raises(ValueError, match="unknown partition scheme"):
        partition(train.y, 10, scheme="shuffled")


def test_more_clients_than_samples_rejected():
    labels = np.arange(5) % NUM_CLASSES
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="without leaving a client empty"):
        partition_iid(5, 10, rng)
    with pytest.raises(ValueError, match="without leaving a client empty"):
        partition_dirichlet(labels, 10, 0.5, rng)


def test_invalid_client_count_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="num_clients must be >= 1"):
        partition_iid(100, 0, rng)
    with pytest.raises(ValueError, match="num_clients must be >= 1"):
        partition_dirichlet(np.zeros(100, dtype=int), 0, 0.5, rng)


def test_invalid_alpha_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="alpha must be > 0"):
        partition_dirichlet(np.zeros(100, dtype=int), 5, 0.0, rng)


def test_partition_summary_reports_per_client_label_counts(fashion):
    train, _ = fashion
    shards = partition(train.y, 5, scheme="dirichlet", alpha=0.5, seed=42)
    summary = partition_summary(train.y, shards)
    assert len(summary) == 5
    assert sum(row["num_examples"] for row in summary) == len(train)
    for row, shard in zip(summary, shards, strict=False):
        assert sum(row["label_counts"]) == shard.size
        assert row["num_classes_present"] == sum(1 for c in row["label_counts"] if c > 0)
