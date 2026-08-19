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


# -- unified loading entry ---------------------------------------------------


def test_load_federated_fashion_matches_direct_calls(fashion):
    from fl.config import DataConfig
    from fl.data import load_federated

    train_direct, _ = fashion
    cfg = DataConfig(dataset="fashion_mnist", num_clients=7, partition="dirichlet")
    train, test, shards = load_federated(cfg, seed=42)
    assert len(train) == len(train_direct)
    assert len(shards) == 7
    direct = partition(train.y, 7, scheme="dirichlet", alpha=0.5, seed=42)
    for a, b in zip(shards, direct, strict=True):
        assert np.array_equal(a, b)
    # The test split is separate and untouched by the partitioner.
    assert len(test) > 0
    assert sum(s.size for s in shards) == len(train)


def test_load_federated_rejects_unknown_dataset():
    from dataclasses import dataclass

    from fl.data import load_federated

    @dataclass
    class Fake:
        dataset: str = "cifar_zzz"
        num_clients: int = 2
        partition: str = "iid"
        dirichlet_alpha: float = 0.5

    with pytest.raises(ValueError, match="unknown dataset"):
        load_federated(Fake(), seed=0)


def test_dataset_num_classes():
    from fl.data import dataset_num_classes

    assert dataset_num_classes("fashion_mnist") == 10
    with pytest.raises(ValueError, match="unknown dataset"):
        dataset_num_classes("nope")


def test_label_entropy():
    from fl.data import label_entropy

    assert label_entropy(np.array([0, 0, 0])) == 0.0  # empty shard
    assert label_entropy(np.array([5, 0, 0])) == 0.0  # single class
    uniform = label_entropy(np.array([10, 10, 10, 10]))
    assert abs(uniform - np.log(4)) < 1e-12  # uniform = log K
    skewed = label_entropy(np.array([97, 1, 1, 1]))
    assert 0.0 < skewed < uniform  # skew strictly between


def test_label_distribution_honours_num_classes():
    labels = np.array([0, 1, 2, 5])
    shard = np.arange(4)
    assert label_distribution(labels, shard, num_classes=62).shape == (62,)
    assert label_distribution(labels, shard, num_classes=62).sum() == 4


# ---------------------------------------------------------------------------
# Paired partitions: per-client TEST shards for personalized evaluation
# ---------------------------------------------------------------------------

#: Synthetic labels, so these run without loading Fashion-MNIST and so the
#: class balance is under the test's control rather than the dataset's.
_PAIR_RNG = np.random.default_rng(11)
PAIR_TRAIN_Y = _PAIR_RNG.integers(0, NUM_CLASSES, size=6_000)
PAIR_TEST_Y = _PAIR_RNG.integers(0, NUM_CLASSES, size=1_000)


class TestPairedPartition:
    """A client's test shard has to come from the client's own distribution.

    Dealing the test split with a fresh Dirichlet draw would give client k a
    training set skewed one way and a test set skewed another, and personalized
    accuracy would then measure the mismatch rather than the personalization.
    """

    @pytest.mark.parametrize("alpha", (0.1, 0.5))
    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_train_half_is_identical_to_the_unpaired_partitioner(self, alpha, seed):
        """So a paired run stays comparable with every Dirichlet run already
        recorded: same seed, same training split, bit for bit."""
        from fl.data import partition_dirichlet_paired

        reference = partition(
            PAIR_TRAIN_Y, num_clients=20, scheme="dirichlet", alpha=alpha, seed=seed
        )
        train_shards, _test_shards = partition_dirichlet_paired(
            PAIR_TRAIN_Y, PAIR_TEST_Y, num_clients=20, alpha=alpha, seed=seed
        )
        for a, b in zip(reference, train_shards, strict=True):
            assert np.array_equal(a, b)

    @pytest.mark.parametrize("alpha", (0.1, 0.5))
    def test_test_shards_are_disjoint_and_exhaustive_over_the_test_split(self, alpha):
        from fl.data import partition_dirichlet_paired

        _train, test_shards = partition_dirichlet_paired(
            PAIR_TRAIN_Y, PAIR_TEST_Y, num_clients=20, alpha=alpha, seed=7
        )
        union = np.concatenate(test_shards)
        assert np.array_equal(np.sort(union), np.arange(PAIR_TEST_Y.size))
        assert np.unique(union).size == union.size
        assert all(s.max(initial=-1) < PAIR_TEST_Y.size for s in test_shards)

    def test_each_client_s_test_labels_follow_its_own_training_labels(self):
        """The property the construction exists to give.

        Compared against the *shuffled* control: the same test shards paired
        with the wrong clients must correlate visibly worse, or the check would
        pass on any partition at all.
        """
        from fl.data import partition_dirichlet_paired

        train_shards, test_shards = partition_dirichlet_paired(
            PAIR_TRAIN_Y, PAIR_TEST_Y, num_clients=20, alpha=0.1, seed=3
        )

        def profile(labels, shard):
            counts = label_distribution(labels, shard, NUM_CLASSES).astype(float)
            return counts / max(1.0, counts.sum())

        matched, mismatched = [], []
        for cid, (tr, te) in enumerate(zip(train_shards, test_shards, strict=True)):
            if te.size < 20:
                continue
            p = profile(PAIR_TRAIN_Y, tr)
            matched.append(float(np.corrcoef(p, profile(PAIR_TEST_Y, te))[0, 1]))
            other = test_shards[(cid + 7) % len(test_shards)]
            if other.size >= 20:
                mismatched.append(float(np.corrcoef(p, profile(PAIR_TEST_Y, other))[0, 1]))

        assert matched, "precondition: some client must have enough held-out data"
        assert np.median(matched) > 0.9
        assert np.median(matched) > np.median(mismatched) + 0.3

    def test_empty_test_shards_are_returned_empty_rather_than_repaired(self):
        """Handing a client with no held-out data one of its neighbour's samples
        would fabricate the quantity being measured. Train shards *are* repaired,
        because a client with no training data breaks round sampling."""
        from fl.data import partition_dirichlet_paired

        train_shards, test_shards = partition_dirichlet_paired(
            PAIR_TRAIN_Y, PAIR_TEST_Y[:40], num_clients=30, alpha=0.05, seed=5
        )
        assert all(s.size > 0 for s in train_shards)
        assert any(s.size == 0 for s in test_shards), "precondition: the split must starve someone"
        assert sum(s.size for s in test_shards) == 40

    def test_iid_pairing_splits_both_sides(self):
        from fl.data import partition_iid_paired

        train_shards, test_shards = partition_iid_paired(6_000, 1_000, 20, seed=4)
        assert sum(s.size for s in train_shards) == 6_000
        assert sum(s.size for s in test_shards) == 1_000
        assert np.unique(np.concatenate(test_shards)).size == 1_000

    def test_invalid_arguments_are_rejected(self):
        from fl.data import partition_dirichlet_paired

        with pytest.raises(ValueError, match="num_clients must be >= 1"):
            partition_dirichlet_paired(PAIR_TRAIN_Y, PAIR_TEST_Y, 0, 0.5)
        with pytest.raises(ValueError, match="alpha must be > 0"):
            partition_dirichlet_paired(PAIR_TRAIN_Y, PAIR_TEST_Y, 10, 0.0)


def test_no_test_image_reaches_a_client_shard_under_the_paired_partition(fashion):
    """Invariant 3, restated for the personalized loader.

    The paired partitioner now touches the test split, so the content-level
    check is repeated against it: hashing pixels proves no held-out image
    reached a training shard even though both splits are now partitioned.
    """
    from fl.data import partition_dirichlet_paired

    train, test = fashion
    train_shards, test_shards = partition_dirichlet_paired(
        train.y, test.y, num_clients=20, alpha=0.1, seed=42
    )
    test_hashes = {hashlib.md5(row.tobytes()).hexdigest() for row in test.x}
    for cid, shard in enumerate(train_shards):
        client_hashes = {hashlib.md5(row.tobytes()).hexdigest() for row in train.take(shard).x}
        assert not client_hashes & test_hashes, f"client {cid} holds held-out images"
    # And the mirror: every test shard indexes the test split and nothing else.
    assert sum(s.size for s in test_shards) == len(test)
    assert max(int(s.max()) for s in test_shards if s.size) < len(test)
