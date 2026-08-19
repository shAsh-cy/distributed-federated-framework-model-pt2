"""FEMNIST loader tests.

Two layers:

* **Unit tests** run against a small synthetic cache built with the same
  :func:`fl.data.pack_femnist` function the real preparation uses, so the pack
  and load logic is covered without the 4 GB download.
* **Integration tests** assert the four partition invariants the natural split
  must satisfy — shards disjoint, union equals the train split, no test sample
  in any client shard, per-writer label distributions measurably non-uniform —
  against the real cache. They skip (with a reason) when the cache has not been
  prepared; CI does not download FEMNIST.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from fl.data import (
    FEMNIST_CACHE,
    FEMNIST_NUM_CLASSES,
    label_distribution,
    label_entropy,
    load_femnist,
    pack_femnist,
)

# -- synthetic cache ---------------------------------------------------------

RNG = np.random.default_rng(7)
WRITERS = 6


def _synthetic_writer(n: int, classes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    x = RNG.random((n, 28, 28)).astype(np.float32)
    y = RNG.choice(classes, size=n)
    return x, y


@pytest.fixture(scope="module")
def synthetic_cache(tmp_path_factory) -> str:
    """A 6-writer cache in the real format, with deliberately skewed labels."""
    train = [_synthetic_writer(10 + 5 * i, tuple(range(i, i + 3))) for i in range(WRITERS)]
    test = [_synthetic_writer(4, tuple(range(i, i + 3))) for i in range(WRITERS)]
    ids = [f"writer_{i}" for i in range(WRITERS)]
    packed = pack_femnist(train, test, ids)
    path = tmp_path_factory.mktemp("femnist") / "cache.npz"
    np.savez_compressed(path, **packed)
    return str(path)


def test_pack_rejects_mismatched_lists():
    with pytest.raises(ValueError, match="parallel"):
        pack_femnist([_synthetic_writer(3, (0,))], [], [])


def test_pack_rejects_disagreeing_sample_counts():
    x, _ = _synthetic_writer(3, (0,))
    with pytest.raises(ValueError, match="disagree"):
        pack_femnist([(x, np.zeros(2))], [(x, np.zeros(3))], ["w"])


def test_load_full_population(synthetic_cache):
    train, test, shards = load_femnist(cache_path=synthetic_cache)
    assert len(shards) == WRITERS
    assert sum(s.size for s in shards) == len(train)
    # Contiguity: shard i is exactly the next block of indices.
    offset = 0
    for s in shards:
        assert np.array_equal(s, np.arange(offset, offset + s.size))
        offset += s.size
    # Normalisation: float32 in [0, 1] with a channel axis.
    assert train.x.dtype == np.float32 and train.x.shape[1:] == (28, 28, 1)
    assert 0.0 <= train.x.min() and train.x.max() <= 1.0
    assert len(test) == WRITERS * 4


def test_subsample_is_seed_deterministic(synthetic_cache):
    a = load_femnist(num_clients=3, seed=11, cache_path=synthetic_cache)
    b = load_femnist(num_clients=3, seed=11, cache_path=synthetic_cache)
    c = load_femnist(num_clients=3, seed=12, cache_path=synthetic_cache)
    assert all(np.array_equal(x, y) for x, y in zip(a[2], b[2], strict=True))
    assert np.array_equal(a[0].y, b[0].y)
    # A different seed picks a different writer subset (overwhelmingly likely
    # with 6C3 = 20 subsets; asserted on data, not chance: labels differ).
    assert not (
        len(c[0]) == len(a[0]) and np.array_equal(np.sort(a[0].y), np.sort(c[0].y))
    ) or not np.array_equal(a[0].y, c[0].y)


def test_subsample_keeps_test_from_same_writers(synthetic_cache):
    _, test, shards = load_femnist(num_clients=2, seed=5, cache_path=synthetic_cache)
    assert len(test) == 2 * 4  # exactly the two selected writers' test samples


def test_subsample_bounds_enforced(synthetic_cache):
    with pytest.raises(ValueError, match=r"num_clients must be in \[1, 6\]"):
        load_femnist(num_clients=7, cache_path=synthetic_cache)
    with pytest.raises(ValueError, match=r"num_clients must be in \[1, 6\]"):
        load_femnist(num_clients=0, cache_path=synthetic_cache)


def test_missing_cache_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_femnist"):
        load_femnist(cache_path=tmp_path / "absent.npz")


# -- real-data integration ---------------------------------------------------

requires_cache = pytest.mark.skipif(
    not FEMNIST_CACHE.is_file(),
    reason="FEMNIST cache not prepared (run scripts/prepare_femnist.py)",
)


@pytest.fixture(scope="module")
def femnist():
    if not FEMNIST_CACHE.is_file():
        pytest.skip("FEMNIST cache not prepared")
    return load_femnist()


@pytest.fixture(scope="module")
def femnist_per_client():
    """The same load, keeping the per-writer test shards personalization needs."""
    from fl.data import load_femnist_per_client

    if not FEMNIST_CACHE.is_file():
        pytest.skip("FEMNIST cache not prepared")
    return load_femnist_per_client()


@requires_cache
def test_real_shards_disjoint_and_union_is_train_split(femnist):
    train, _test, shards = femnist
    seen = np.concatenate(shards)
    assert seen.size == len(train)  # exhaustive
    assert np.unique(seen).size == seen.size  # pairwise disjoint
    assert np.array_equal(np.sort(seen), np.arange(len(train)))  # exactly the split


@requires_cache
def test_real_no_test_sample_in_any_client_shard(femnist):
    train, test, shards = femnist
    # Structural, and strict: shards index the train arrays only, and the test
    # split is a separate object -- no shard index can address a test sample.
    top = max(int(s.max()) for s in shards)
    assert top < len(train)
    assert all(int(s.min()) >= 0 for s in shards)
    assert len(test) > 0 and test.x.shape[0] == test.y.shape[0]


@requires_cache
def test_real_cross_split_byte_collisions_are_the_upstream_rate(femnist):
    """Content check, bounded at the dataset's own duplicate rate.

    Measured on the real cache (2026-07-31): 649 of 77,483 test images (0.84%)
    are byte-identical to some train image -- every one label-consistent, with
    a mean ink fraction of 3.9%, i.e. minimal-stroke glyphs that collide after
    uint8 quantisation. The train split *internally* contains 2,732 byte
    duplicates, so byte-identity is a property of LEAF/TFF's upstream
    preprocessing, not of this repo's packing: the writer-level train/test
    split is taken from upstream verbatim and no sample is moved between
    splits here. This test pins that rate so a packing regression (which would
    send it far higher) still fails loudly.
    """
    train, test, _shards = femnist
    train_hashes = {hashlib.sha1(im.tobytes()).digest() for im in train.x}
    collisions = sum(1 for im in test.x if hashlib.sha1(im.tobytes()).digest() in train_hashes)
    assert collisions / len(test.x) < 0.01, collisions


@requires_cache
def test_real_writer_label_distributions_measurably_non_uniform(femnist):
    train, _test, shards = femnist
    pooled = label_entropy(label_distribution(train.y, np.arange(len(train)), FEMNIST_NUM_CLASSES))
    per_writer = [
        label_entropy(label_distribution(train.y, s, FEMNIST_NUM_CLASSES)) for s in shards
    ]
    mean_writer = float(np.mean(per_writer))
    # Real writers are skewed: mean per-writer entropy must sit clearly below
    # the pooled entropy. The margin is deliberately loose; the measured values
    # are reported in docs/femnist_cohort.md.
    assert mean_writer < pooled - 0.05, (mean_writer, pooled)


@requires_cache
def test_real_population_shape(femnist):
    train, test, shards = femnist
    assert len(shards) == 3400
    assert train.y.max() < FEMNIST_NUM_CLASSES
    assert test.y.max() < FEMNIST_NUM_CLASSES
    assert all(s.size > 0 for s in shards)


# -- per-writer test shards (personalized evaluation) ------------------------


class TestPerClientTestShards:
    """FEMNIST's answer to the hardest part of personalized evaluation.

    Per-client accuracy needs each client's own held-out data. On a pooled
    dataset that has to be constructed; here it is taken from LEAF's by-writer
    split verbatim, so ``test_shards[i]`` is writer ``i``'s own samples as a
    matter of provenance and not of construction. These tests hold that
    provenance to account.
    """

    def test_test_shards_are_contiguous_disjoint_and_exhaustive(self, synthetic_cache):
        from fl.data import load_femnist_per_client

        _train, test, _shards, test_shards = load_femnist_per_client(cache_path=synthetic_cache)
        assert len(test_shards) == WRITERS
        offset = 0
        for shard in test_shards:
            assert np.array_equal(shard, np.arange(offset, offset + shard.size))
            offset += shard.size
        assert offset == len(test)
        union = np.concatenate(test_shards)
        assert np.unique(union).size == union.size == len(test)

    def test_a_writer_s_test_labels_come_from_that_writer_s_own_classes(self, synthetic_cache):
        """The pairing check. The fixture gives writer ``i`` classes
        ``{i, i+1, i+2}`` in *both* splits, so a shard list that paired writers
        with the wrong test blocks would show up as an out-of-range label."""
        from fl.data import load_femnist_per_client

        train, test, shards, test_shards = load_femnist_per_client(cache_path=synthetic_cache)
        for i, (tr, te) in enumerate(zip(shards, test_shards, strict=True)):
            assert set(np.unique(test.y[te])) <= set(range(i, i + 3))
            assert set(np.unique(test.y[te])) <= set(np.unique(train.y[tr])) | set(range(i, i + 3))

    def test_the_three_value_loader_is_the_same_split_without_the_fourth(self, synthetic_cache):
        from fl.data import load_femnist, load_femnist_per_client

        three = load_femnist(num_clients=4, seed=9, cache_path=synthetic_cache)
        four = load_femnist_per_client(num_clients=4, seed=9, cache_path=synthetic_cache)
        assert len(three) == 3 and len(four) == 4
        assert np.array_equal(three[0].y, four[0].y)
        assert np.array_equal(three[1].y, four[1].y)
        assert all(np.array_equal(a, b) for a, b in zip(three[2], four[2], strict=True))

    def test_a_subsampled_population_keeps_its_own_writers_test_data(self, synthetic_cache):
        from fl.data import load_femnist_per_client

        _train, test, shards, test_shards = load_femnist_per_client(
            num_clients=2, seed=5, cache_path=synthetic_cache
        )
        assert len(shards) == len(test_shards) == 2
        assert sum(s.size for s in test_shards) == len(test) == 2 * 4


@requires_cache
def test_real_test_shards_partition_the_real_test_split(femnist_per_client):
    train, test, shards, test_shards = femnist_per_client
    assert len(shards) == len(test_shards) == 3400
    union = np.concatenate(test_shards)
    assert np.array_equal(np.sort(union), np.arange(len(test)))
    assert all(s.size > 0 for s in test_shards), "a writer with no held-out data is unevaluable"
    assert max(int(s.max()) for s in test_shards) < len(test)
    assert max(int(s.max()) for s in shards) < len(train)


@requires_cache
def test_real_writers_test_labels_match_their_own_training_labels(femnist_per_client):
    """Personalization's premise, measured rather than assumed.

    A writer's held-out label profile must resemble that writer's training
    profile far more than it resembles another writer's, or there would be
    nothing for a local head to specialise to. The mismatched control is what
    makes the number mean something.
    """
    train, test, shards, test_shards = femnist_per_client

    def profile(labels, shard):
        counts = label_distribution(labels, shard, FEMNIST_NUM_CLASSES).astype(float)
        return counts / max(1.0, counts.sum())

    matched, mismatched = [], []
    for cid in range(0, 3400, 17):  # a spread sample; all 3,400 would be slow
        p = profile(train.y, shards[cid])
        matched.append(float(np.corrcoef(p, profile(test.y, test_shards[cid]))[0, 1]))
        other = (cid + 811) % 3400
        mismatched.append(float(np.corrcoef(p, profile(test.y, test_shards[other]))[0, 1]))

    assert np.median(matched) > np.median(mismatched) + 0.1, (
        np.median(matched),
        np.median(mismatched),
    )
