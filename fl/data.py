"""Fashion-MNIST loading and client partitioning.

Two invariants hold for every partition this module produces, and both are tested:

1. **Shards are disjoint and exhaustive.** Every training index lands in exactly
   one client. No sample is duplicated across clients, and none is dropped.
2. **The test set is never partitioned for training.** :func:`load_fashion_mnist`
   returns it separately and only the server ever touches it. Clients receive
   training indices only, so a client physically cannot evaluate on held-out
   data.

   Personalized evaluation adds *test shards* -- ``test_shards[k]`` is client
   ``k``'s own held-out data, from :func:`load_femnist_per_client` (real, from
   LEAF's by-writer split) or :func:`partition_dirichlet_paired` (synthetic, for
   a pooled dataset). This does not weaken the invariant and the wording above is
   precise about which half changed: test shards index the *test* arrays, train
   shards index the *train* arrays, the two arrays are separate objects, and no
   function here ever hands a test shard to a training path. What personalized
   evaluation needs is the ability to *score* client ``k`` on client ``k``'s own
   data, which is a property of the evaluator, not of what the client holds.

Both an IID and a non-IID split are provided. The non-IID split is the default,
because an IID split makes every client's gradient an unbiased estimate of the
same global gradient and FedAvg collapses into mildly noisy centralised SGD --
the client-drift problem that motivates the whole field simply does not appear.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

NUM_CLASSES: int = 10


@dataclass(frozen=True)
class Dataset:
    """An in-memory image classification dataset."""

    x: np.ndarray  # float32, (n, 28, 28, 1), scaled to [0, 1]
    y: np.ndarray  # int64, (n,)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def take(self, indices: np.ndarray) -> Dataset:
        """Return the sub-dataset at ``indices`` (a copy, not a view)."""
        idx = np.asarray(indices, dtype=np.int64)
        return Dataset(x=self.x[idx].copy(), y=self.y[idx].copy())


def _prepare(x: np.ndarray, y: np.ndarray) -> Dataset:
    x = x.astype("float32") / 255.0
    if x.ndim == 3:
        x = np.expand_dims(x, -1)
    return Dataset(x=x, y=y.astype("int64").reshape(-1))


def load_fashion_mnist() -> tuple[Dataset, Dataset]:
    """Load Fashion-MNIST as ``(train, test)``.

    The test split is returned as a separate object and is never passed to the
    partitioner. The server holds it; clients never see it.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    return _prepare(x_train, y_train), _prepare(x_test, y_test)


#: Classes per dataset. The model output size and every label histogram are
#: derived from this, never hard-coded at a call site.
DATASET_NUM_CLASSES: dict[str, int] = {"fashion_mnist": 10, "femnist": 62}

FEMNIST_NUM_CLASSES: int = 62

#: Where the packed FEMNIST cache lives, relative to the repo root (data/ is
#: gitignored). Produced once by ``scripts/prepare_femnist.py``.
FEMNIST_CACHE: Path = Path("data/femnist/femnist62.npz")

#: Writers in the full LEAF-derived federated EMNIST train split.
FEMNIST_TOTAL_WRITERS: int = 3400


def dataset_num_classes(dataset: str) -> int:
    """Number of classes for a dataset name from config."""
    try:
        return DATASET_NUM_CLASSES[dataset]
    except KeyError:
        raise ValueError(
            f"unknown dataset {dataset!r}; available: {sorted(DATASET_NUM_CLASSES)}"
        ) from None


def pack_femnist(
    train_per_writer: list[tuple[np.ndarray, np.ndarray]],
    test_per_writer: list[tuple[np.ndarray, np.ndarray]],
    writer_ids: list[str],
) -> dict[str, np.ndarray]:
    """Pack per-writer (images, labels) pairs into the flat cache format.

    Images are stored uint8 (pixels * 255) at a quarter of the float32 size;
    :func:`_prepare` restores [0, 1] float32 at load time. Writer boundaries are
    stored as offsets: writer ``i``'s train samples occupy
    ``[train_starts[i], train_starts[i+1])``, which makes the natural shards
    disjoint and exhaustive *by construction* rather than by bookkeeping.
    """
    if not (len(train_per_writer) == len(test_per_writer) == len(writer_ids)):
        raise ValueError("train, test and writer-id lists must be parallel")

    def flatten(per_writer: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, ...]:
        xs, ys, starts = [], [], [0]
        for x, y in per_writer:
            x = np.asarray(x)
            y = np.asarray(y).reshape(-1)
            if x.shape[0] != y.shape[0]:
                raise ValueError("images and labels disagree on sample count")
            if x.dtype != np.uint8:
                x = np.round(np.asarray(x, dtype=np.float64) * 255.0).astype(np.uint8)
            xs.append(x.reshape(-1, 28, 28))
            ys.append(y.astype(np.uint8))
            starts.append(starts[-1] + int(y.shape[0]))
        return (
            np.concatenate(xs) if xs else np.empty((0, 28, 28), np.uint8),
            np.concatenate(ys) if ys else np.empty((0,), np.uint8),
            np.asarray(starts, dtype=np.int64),
        )

    train_x, train_y, train_starts = flatten(train_per_writer)
    test_x, test_y, test_starts = flatten(test_per_writer)
    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_starts": train_starts,
        "test_x": test_x,
        "test_y": test_y,
        "test_starts": test_starts,
        "writer_ids": np.asarray(writer_ids, dtype=np.str_),
    }


def prepare_femnist_cache(
    cache_path: str | Path = FEMNIST_CACHE,
    download_dir: str | Path = "data/tff_cache",
) -> Path:
    """One-time: download federated EMNIST and pack it to the local cache.

    Uses ``tff.simulation.datasets.emnist`` with ``only_digits=False`` — the
    62-class, by-writer federated EMNIST that TFF documents as derived from the
    LEAF benchmark's preprocessing. 3,400 writers. The download (~170 MB
    compressed, ~4.4 GB unpacked) and this packing step both land under the
    gitignored ``data/`` directory.
    """
    cache_path = Path(cache_path)
    if cache_path.is_file():
        return cache_path

    import tensorflow_federated as tff  # pragma: no cover - exercised via scripts

    train_cd, test_cd = tff.simulation.datasets.emnist.load_data(  # pragma: no cover
        only_digits=False, cache_dir=str(download_dir)
    )
    ids = sorted(train_cd.client_ids)  # pragma: no cover
    if sorted(test_cd.client_ids) != ids:  # pragma: no cover
        raise RuntimeError("train and test writer sets differ; refusing to pack")

    def pull(cd, cid: str) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        ds = cd.create_tf_dataset_for_client(cid).batch(1 << 14)
        xs, ys = [], []
        for batch in ds:
            xs.append(batch["pixels"].numpy())
            ys.append(batch["label"].numpy())
        return np.concatenate(xs), np.concatenate(ys)

    packed = pack_femnist(  # pragma: no cover
        [pull(train_cd, cid) for cid in ids],
        [pull(test_cd, cid) for cid in ids],
        ids,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)  # pragma: no cover
    np.savez_compressed(cache_path, **packed)  # pragma: no cover
    return cache_path  # pragma: no cover


def load_femnist_per_client(
    num_clients: int | None = None,
    seed: int = 42,
    cache_path: str | Path = FEMNIST_CACHE,
) -> tuple[Dataset, Dataset, list[np.ndarray], list[np.ndarray]]:
    """Load FEMNIST with its natural by-writer partition, train *and* test.

    The fourth return value is what personalized evaluation needs and what a
    pooled test split cannot provide: writer ``i``'s own held-out samples,
    ``test_shards[i]``, indexing ``test``. LEAF's federated EMNIST carries this
    natively -- the by-writer train/test split is upstream's, taken verbatim --
    so per-client accuracy on FEMNIST is a measurement, not a construction.

    Args:
        num_clients: Writers to include, drawn as a seeded uniform subsample of
            the full writer set (``None`` or the full count selects everyone).
            The test set is the pooled test split of the *same* writers, so a
            subsampled population is evaluated on its own writers' held-out
            samples, not on writers it never trained with.
        seed: Seed for the writer subsample only.
        cache_path: Cache produced by :func:`prepare_femnist_cache`.

    Returns:
        ``(train, test, shards, test_shards)`` — both shard lists are contiguous
        index ranges into their own split, one entry per selected writer,
        disjoint and exhaustive by construction. ``shards[i]`` and
        ``test_shards[i]`` are the same writer; nothing pairs them up after the
        fact, they are built in one pass over the same writer order.
    """
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"FEMNIST cache not found at {cache_path}. Run scripts/prepare_femnist.py "
            "once (downloads ~170 MB, packs to ~data/femnist/) before using "
            "dataset 'femnist'."
        )
    with np.load(cache_path, allow_pickle=False) as z:
        train_x, train_y = z["train_x"], z["train_y"]
        train_starts = z["train_starts"]
        test_x, test_y = z["test_x"], z["test_y"]
        test_starts = z["test_starts"]
        total = len(z["writer_ids"])

    if num_clients is None:
        num_clients = total
    if not 1 <= num_clients <= total:
        raise ValueError(f"num_clients must be in [1, {total}], got {num_clients}")

    if num_clients == total:
        selected = np.arange(total)
    else:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(total, size=num_clients, replace=False))

    shards: list[np.ndarray] = []
    test_shards: list[np.ndarray] = []
    train_pieces_x, train_pieces_y, test_pieces_x, test_pieces_y = [], [], [], []
    offset = 0
    test_offset = 0
    for w in selected:
        lo, hi = int(train_starts[w]), int(train_starts[w + 1])
        train_pieces_x.append(train_x[lo:hi])
        train_pieces_y.append(train_y[lo:hi])
        shards.append(np.arange(offset, offset + (hi - lo), dtype=np.int64))
        offset += hi - lo
        tlo, thi = int(test_starts[w]), int(test_starts[w + 1])
        test_pieces_x.append(test_x[tlo:thi])
        test_pieces_y.append(test_y[tlo:thi])
        test_shards.append(np.arange(test_offset, test_offset + (thi - tlo), dtype=np.int64))
        test_offset += thi - tlo

    train = _prepare(np.concatenate(train_pieces_x), np.concatenate(train_pieces_y))
    test = _prepare(np.concatenate(test_pieces_x), np.concatenate(test_pieces_y))
    return train, test, shards, test_shards


def load_femnist(
    num_clients: int | None = None,
    seed: int = 42,
    cache_path: str | Path = FEMNIST_CACHE,
) -> tuple[Dataset, Dataset, list[np.ndarray]]:
    """:func:`load_femnist_per_client` without the per-writer test shards.

    The three-value form every non-personalized caller uses: the test split is
    pooled and held by the server, which is all a global-accuracy figure needs.
    """
    train, test, shards, _test_shards = load_femnist_per_client(
        num_clients=num_clients, seed=seed, cache_path=cache_path
    )
    return train, test, shards


def load_federated(data_cfg, seed: int = 42) -> tuple[Dataset, Dataset, list[np.ndarray]]:
    """Load a dataset and its client partition from a :class:`fl.config.DataConfig`.

    The single entry point every runner uses, so switching dataset is a config
    change and nothing else. Returns ``(train, test, shards)`` where ``shards``
    is one sorted index array into ``train`` per client.

    For ``fashion_mnist`` the partition is synthetic (IID or Dirichlet
    label-skew over a pooled dataset). For ``femnist`` it is natural: each
    shard is one real writer's samples, and ``num_clients`` selects how many
    writers form the population.
    """
    if data_cfg.dataset == "fashion_mnist":
        train, test = load_fashion_mnist()
        shards = partition(
            train.y,
            num_clients=data_cfg.num_clients,
            scheme=data_cfg.partition,
            alpha=data_cfg.dirichlet_alpha,
            seed=seed,
        )
        return train, test, shards
    if data_cfg.dataset == "femnist":
        return load_femnist(num_clients=data_cfg.num_clients, seed=seed)
    raise ValueError(
        f"unknown dataset {data_cfg.dataset!r}; available: {sorted(DATASET_NUM_CLASSES)}"
    )


def load_federated_per_client(
    data_cfg, seed: int = 42
) -> tuple[Dataset, Dataset, list[np.ndarray], list[np.ndarray]]:
    """:func:`load_federated` plus per-client test shards, for personalized eval.

    The fourth return value is the whole point and its provenance differs by
    dataset, which the write-up must not blur:

    * ``femnist`` — **real**. Each client is one writer and ``test_shards[k]`` is
      that writer's own held-out samples, taken verbatim from LEAF's by-writer
      split. Nothing here constructs the client-test correspondence; it is
      upstream's.
    * ``fashion_mnist`` — **synthetic**, via :func:`partition_dirichlet_paired`
      (or :func:`partition_iid_paired`): the test split is dealt across clients
      with the *same* per-class proportions as the training split, so client
      ``k``'s test distribution matches its train distribution by construction.
      That is a modelling choice, not a measurement, and a personalization result
      on Fashion-MNIST is a result about that construction.

    Both shard lists index their own split only. Test shards may be empty at
    small ``alpha``; they are returned empty rather than repaired, and the caller
    reports how many clients that leaves unevaluable.
    """
    if data_cfg.dataset == "fashion_mnist":
        train, test = load_fashion_mnist()
        if data_cfg.partition == "iid":
            shards, test_shards = partition_iid_paired(
                len(train), len(test), data_cfg.num_clients, seed=seed
            )
        else:
            shards, test_shards = partition_dirichlet_paired(
                train.y, test.y, data_cfg.num_clients, data_cfg.dirichlet_alpha, seed=seed
            )
        return train, test, shards, test_shards
    if data_cfg.dataset == "femnist":
        return load_femnist_per_client(num_clients=data_cfg.num_clients, seed=seed)
    raise ValueError(
        f"unknown dataset {data_cfg.dataset!r}; available: {sorted(DATASET_NUM_CLASSES)}"
    )


def partition_iid(
    num_examples: int, num_clients: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Split indices uniformly at random.

    Every client's shard is an unbiased sample of the global distribution, so all
    label histograms match up to sampling noise.
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    if num_examples < num_clients:
        raise ValueError(
            f"cannot split {num_examples} examples across {num_clients} clients "
            "without leaving a client empty"
        )
    indices = rng.permutation(num_examples)
    # array_split handles the remainder: the first (n % k) shards get one extra.
    return [np.sort(shard) for shard in np.array_split(indices, num_clients)]


def partition_dirichlet(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Split indices with a per-client skewed label distribution.

    For each class *c*, proportions ``p ~ Dir(alpha * 1_K)`` are drawn over the
    ``K`` clients and class *c*'s samples are dealt out in those proportions. This
    is the standard label-skew construction used in the FL literature.

    ``alpha`` controls skew: as ``alpha -> inf`` the split tends to IID; as
    ``alpha -> 0`` each class concentrates on a single client. ``alpha = 0.5``
    gives clients that are visibly skewed but still see most classes.

    Because every class's indices are dealt out exhaustively, the union of shards
    is the complete index set and shards are pairwise disjoint by construction.

    A client that draws nothing at all would break the round-sampling logic, so
    empty shards are repaired by moving one sample from the largest shard. With
    the default alpha this effectively never fires; at very small alpha it does.
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] < num_clients:
        raise ValueError(
            f"cannot split {labels.shape[0]} examples across {num_clients} clients "
            "without leaving a client empty"
        )

    shards, _proportions = _deal_dirichlet(labels, num_clients, alpha, rng)
    return _repair_empty_shards(shards)


def _deal_dirichlet(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
    proportions: dict[int, np.ndarray] | None = None,
) -> tuple[list[np.ndarray], dict[int, np.ndarray]]:
    """Deal each class's indices across clients in Dirichlet proportions.

    Returns the raw (unrepaired) shards *and* the per-class proportion vectors
    that produced them. Handing the proportions back is what lets a second split
    -- a test split, in :func:`partition_dirichlet_paired` -- be dealt in the
    same shape as the first instead of drawing its own, which would give each
    client a test set from a different distribution than its training set and
    make per-client accuracy unreadable.

    When ``proportions`` is supplied nothing is drawn from ``rng`` except the
    within-class shuffles, so the caller's stream is untouched by this choice.
    """
    labels = np.asarray(labels).reshape(-1)
    drawn: dict[int, np.ndarray] = {}
    pieces: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        given = None if proportions is None else proportions.get(int(cls))
        p = rng.dirichlet(np.repeat(alpha, num_clients)) if given is None else np.asarray(given)
        if p.shape != (num_clients,):
            raise ValueError(
                f"proportions for class {int(cls)} have shape {p.shape}, expected ({num_clients},)"
            )
        drawn[int(cls)] = p
        # Cut points partition cls_idx into exactly num_clients contiguous pieces.
        cuts = (np.cumsum(p)[:-1] * len(cls_idx)).astype(int)
        for cid, piece in enumerate(np.split(cls_idx, cuts)):
            if piece.size:
                pieces[cid].append(piece)

    out = [
        np.sort(np.concatenate(chunks)) if chunks else np.empty(0, dtype=np.int64)
        for chunks in pieces
    ]
    return out, drawn


def partition_dirichlet_paired(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split *both* splits with one set of per-class client proportions.

    Personalized evaluation needs each client's own held-out data. FEMNIST has
    it natively (:func:`load_femnist_per_client`); a pooled dataset does not, and
    inventing it carelessly destroys the measurement. Dealing the test set with
    *freshly drawn* proportions would give client ``k`` a training set skewed one
    way and a test set skewed another, so a personalized head fitted to the first
    would be scored against the second and personalization would look harmful for
    reasons that have nothing to do with personalization. Reusing the training
    split's proportions is the construction that makes the two match.

    Two properties this deliberately keeps:

    * The train half is **bit-identical** to ``partition_dirichlet`` at the same
      seed -- the proportions are drawn from the same stream in the same order,
      and the test deal runs on its own generator. A Dirichlet run here is
      therefore comparable to every Dirichlet run already recorded.
    * Test shards are **not** repaired to be non-empty. Handing a client with no
      held-out data one sample from its neighbour would fabricate exactly the
      quantity being measured; such clients are returned empty, and the caller
      counts and excludes them (and says so).
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    raw_train, proportions = _deal_dirichlet(
        train_labels, num_clients, alpha, np.random.default_rng(seed)
    )
    train_shards = _repair_empty_shards(raw_train)
    test_shards, _ = _deal_dirichlet(
        test_labels,
        num_clients,
        alpha,
        np.random.default_rng(seed + 1),
        proportions=proportions,
    )
    return train_shards, test_shards


def partition_iid_paired(
    num_train: int,
    num_test: int,
    num_clients: int,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """IID counterpart of :func:`partition_dirichlet_paired`.

    Kept because it is the control that says what the measurement reads when
    there is nothing to personalize: every client's test shard is an unbiased
    sample of the same distribution, so a personalized head has no client-specific
    structure to exploit and the two arms should coincide.
    """
    train = partition_iid(num_train, num_clients, np.random.default_rng(seed))
    test = partition_iid(num_test, num_clients, np.random.default_rng(seed + 1))
    return train, test


def _repair_empty_shards(shards: list[np.ndarray]) -> list[np.ndarray]:
    """Ensure no shard is empty, preserving disjointness and exhaustiveness."""
    for cid, shard in enumerate(shards):
        if shard.size:
            continue
        donor = int(np.argmax([s.size for s in shards]))
        if shards[donor].size < 2:
            raise ValueError("not enough samples to give every client a non-empty shard")
        shards[cid] = shards[donor][-1:].copy()
        shards[donor] = shards[donor][:-1]
    return [np.sort(s) for s in shards]


def partition(
    labels: np.ndarray,
    num_clients: int,
    scheme: str = "dirichlet",
    alpha: float = 0.5,
    seed: int = 42,
) -> list[np.ndarray]:
    """Partition training indices across clients.

    Args:
        labels: Training labels; only their length is used for the IID scheme.
        num_clients: Number of shards to produce.
        scheme: ``"iid"`` or ``"dirichlet"``.
        alpha: Dirichlet concentration; ignored when ``scheme == "iid"``.
        seed: Seed for the partition RNG, independent of model initialisation.

    Returns:
        One sorted index array per client.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).reshape(-1)
    if scheme == "iid":
        return partition_iid(labels.shape[0], num_clients, rng)
    if scheme == "dirichlet":
        return partition_dirichlet(labels, num_clients, alpha, rng)
    raise ValueError(f"unknown partition scheme {scheme!r}; expected 'iid' or 'dirichlet'")


def label_distribution(
    labels: np.ndarray, shard: np.ndarray, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Per-class counts for one shard, length ``num_classes``."""
    return np.bincount(np.asarray(labels).reshape(-1)[shard], minlength=num_classes)


def label_entropy(counts: np.ndarray) -> float:
    """Shannon entropy (nats) of a label-count vector; empty counts give 0."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def partition_summary(
    labels: np.ndarray, shards: list[np.ndarray], num_classes: int = NUM_CLASSES
) -> list[dict]:
    """Human-readable per-client summary, logged by the server at startup."""
    return [
        {
            "client_index": cid,
            "num_examples": int(shard.size),
            "label_counts": label_distribution(labels, shard, num_classes).tolist(),
            "num_classes_present": int((label_distribution(labels, shard, num_classes) > 0).sum()),
        }
        for cid, shard in enumerate(shards)
    ]
