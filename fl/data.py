"""Fashion-MNIST loading and client partitioning.

Two invariants hold for every partition this module produces, and both are tested:

1. **Shards are disjoint and exhaustive.** Every training index lands in exactly
   one client. No sample is duplicated across clients, and none is dropped.
2. **The test set is never partitioned.** :func:`load_fashion_mnist` returns it
   separately and only the server ever touches it. Clients receive training
   indices only, so a client physically cannot evaluate on held-out data.

Both an IID and a non-IID split are provided. The non-IID split is the default,
because an IID split makes every client's gradient an unbiased estimate of the
same global gradient and FedAvg collapses into mildly noisy centralised SGD --
the client-drift problem that motivates the whole field simply does not appear.
"""

from __future__ import annotations

from dataclasses import dataclass

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
DATASET_NUM_CLASSES: dict[str, int] = {"fashion_mnist": 10}


def dataset_num_classes(dataset: str) -> int:
    """Number of classes for a dataset name from config."""
    try:
        return DATASET_NUM_CLASSES[dataset]
    except KeyError:
        raise ValueError(
            f"unknown dataset {dataset!r}; available: {sorted(DATASET_NUM_CLASSES)}"
        ) from None


def load_federated(data_cfg, seed: int = 42) -> tuple[Dataset, Dataset, list[np.ndarray]]:
    """Load a dataset and its client partition from a :class:`fl.config.DataConfig`.

    The single entry point every runner uses, so switching dataset is a config
    change and nothing else. Returns ``(train, test, shards)`` where ``shards``
    is one sorted index array into ``train`` per client.

    For ``fashion_mnist`` the partition is synthetic (IID or Dirichlet
    label-skew over a pooled dataset).
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

    shards: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Cut points partition cls_idx into exactly num_clients contiguous pieces.
        cuts = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
        for cid, piece in enumerate(np.split(cls_idx, cuts)):
            if piece.size:
                shards[cid].append(piece)

    out = [
        np.sort(np.concatenate(pieces)) if pieces else np.empty(0, dtype=np.int64)
        for pieces in shards
    ]
    return _repair_empty_shards(out)


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
