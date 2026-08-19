"""FedRep-style personalization: a globally aggregated backbone, a local head.

The setting
-----------
A global model is a single answer to a question every client asks differently.
Under label skew that answer is a compromise, and the clients it fits worst are
the ones whose data least resembles the population -- which is to say, the
clients federated learning exists to serve. Personalization keeps one model per
client without giving up the shared statistical strength: the *representation*
(everything below the classifier) is federated as usual, and the *head* is
fitted locally and never leaves the device.

FedRep (Collins et al., ICML 2021) is the version implemented here: each round a
sampled client fixes the received representation and takes a few local steps on
its head, then fixes the head and takes local steps on the representation, and
submits the representation alone. The head persists on the client across rounds.

What this module owns, and what it does not
-------------------------------------------
This module is framework-neutral. It owns the *state* personalization adds --
:class:`HeadStore`, the per-client head keyed by client id -- and the reporting
that makes a personalization result readable: a distribution rather than a mean
(:func:`distribution_summary`), a paired per-client comparison
(:func:`paired_delta_summary`), and the communication arithmetic
(:func:`wire_saving`). The split itself is :mod:`fl.archspec`'s
``personal_layers`` marker; converting a model to and from it is
:mod:`fl.adapters`; keeping the head off the wire is :mod:`fl.serialization`.
The training loop that uses all four lives in
``scripts/personalization_experiments.py``.

Scope, stated once
------------------
Personalization runs on the **in-process harness**, not on the gRPC container
path. The reason is not effort: a personalized gRPC client must persist its head
across process restarts and re-registrations, which is client-side durable state
the current client (``fl/client.py``) does not have and cannot fake -- a client
that reclaims its shard after a restart would otherwise resume with a head that
is either another client's or freshly initialised, and the second is worse than
it sounds because it is invisible in the metrics. Every recorded personalization
number is therefore a harness number, exactly as the FEMNIST results already
are. The wire-format support here (backbone-only encode, head-tensor rejection
on decode) is what a gRPC implementation would build on, and it is tested, but
it is not wired into a deployed path and this documentation does not pretend
otherwise.

Why the distribution, not the mean
----------------------------------
The mean over clients is the one statistic that cannot show what personalization
does. A method that lifts the median client by a point and the worst decile by
fifteen, and a method that lifts everyone by two, report the same mean and are
not the same result. Every figure this module produces is therefore reported per
client -- full arrays in the JSON, deciles and quartiles in the summary -- so the
tail is visible in the record and not only in the headline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Weights = list[np.ndarray]


class PersonalizationError(ValueError):
    """Raised when personal state does not match the spec it belongs to."""


class HeadStore:
    """Per-client classifier heads, keyed by client id, persisting across rounds.

    Copies on the way in and on the way out, deliberately. Handing out the
    stored array would let a caller mutate one client's head through a
    reference obtained while training another -- the exact cross-client leak
    personalization must not have, and one no accuracy figure would reveal. The
    copy costs 32 KB per client on ``femnist_cnn``; the alternative costs
    correctness.

    A client that has never trained reads back ``initial_head``: the head the
    global model was initialised with. That is the honest cold-start state, and
    :meth:`updates` reports how many rounds each client has actually trained, so
    never-sampled clients can be counted and reported separately rather than
    silently averaged in as though they had personalized.
    """

    def __init__(self, spec, initial_head: Weights) -> None:
        self.spec = spec
        self._expected = spec.personal_shapes()
        self._validate(initial_head, "initial head")
        self._initial: Weights = [np.array(w, dtype=np.float32, copy=True) for w in initial_head]
        self._heads: dict[str, Weights] = {}
        self._updates: dict[str, int] = {}

    def _validate(self, head: Weights, what: str) -> None:
        got = [tuple(np.shape(w)) for w in head]
        if got != self._expected:
            raise PersonalizationError(
                f"{what} does not match the head of spec {self.spec.name!r}: "
                f"got {got}, expected {self._expected}"
            )

    def get(self, client_id) -> Weights:
        """This client's head, or the initial head if it has never trained."""
        stored = self._heads.get(str(client_id))
        source = self._initial if stored is None else stored
        return [np.array(w, dtype=np.float32, copy=True) for w in source]

    def put(self, client_id, head: Weights) -> None:
        """Record this client's head after local training."""
        self._validate(head, f"head for client {client_id!r}")
        key = str(client_id)
        self._heads[key] = [np.array(w, dtype=np.float32, copy=True) for w in head]
        self._updates[key] = self._updates.get(key, 0) + 1

    def updates(self, client_id) -> int:
        """Number of rounds this client has trained its head."""
        return self._updates.get(str(client_id), 0)

    def has_trained(self, client_id) -> bool:
        return str(client_id) in self._heads

    @property
    def num_trained(self) -> int:
        return len(self._heads)

    def participation(self, client_ids) -> dict[str, int]:
        """Per-client head-update counts, including the zeros."""
        return {str(cid): self.updates(cid) for cid in client_ids}


def distribution_summary(values, quantile: float = 0.1) -> dict:
    """Summarise a per-client metric as a distribution, tails first.

    ``worst_decile_mean`` is the mean over the lowest ``quantile`` fraction of
    clients (at least one client), not the value at the 10th percentile: it
    answers "how are the clients this model serves worst actually doing", which
    a single percentile point does not.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("distribution_summary needs at least one value")
    k = max(1, int(np.floor(quantile * arr.size)))
    order = np.sort(arr)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(order[0]),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(order[-1]),
        "decile_fraction": float(quantile),
        "decile_size": int(k),
        "worst_decile_mean": float(order[:k].mean()),
        "best_decile_mean": float(order[-k:].mean()),
    }


def paired_delta_summary(baseline, personalized, quantile: float = 0.1) -> dict:
    """Per-client change, paired client by client.

    The pairing is what makes this stronger than differencing two summaries: the
    same client appears on both sides, so ``fraction_improved`` and the worst
    decile *of the deltas* are statements about individual clients rather than
    about two populations that happen to share a size. The worst decile of the
    delta distribution is where a method that helps on average while actively
    hurting a minority becomes visible.
    """
    a = np.asarray(list(baseline), dtype=np.float64)
    b = np.asarray(list(personalized), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"paired summaries need matching client counts, got {a.shape} vs {b.shape}"
        )
    delta = b - a
    summary = distribution_summary(delta, quantile=quantile)
    summary.update(
        {
            "fraction_improved": float(np.mean(delta > 0)),
            "fraction_unchanged": float(np.mean(delta == 0)),
            "fraction_worsened": float(np.mean(delta < 0)),
            "baseline_mean": float(a.mean()),
            "personalized_mean": float(b.mean()),
        }
    )
    return summary


def weighted_mean(values, weights) -> float:
    """Sample-count-weighted mean of a per-client metric.

    Reported alongside the unweighted mean because they answer different
    questions: weighted is "accuracy over the pooled test set", unweighted is
    "accuracy for the average client". Under heterogeneity these separate, and
    which one a claim is about should never have to be inferred.
    """
    v = np.asarray(list(values), dtype=np.float64)
    w = np.asarray(list(weights), dtype=np.float64)
    if v.shape != w.shape:
        raise ValueError(f"values and weights must be parallel, got {v.shape} vs {w.shape}")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return float((v * w).sum() / total)


def paired_profile_alignment(
    train_labels, shards, test_labels, test_shards, num_classes: int
) -> np.ndarray:
    """Per client: how much better its held-out labels match *its own* training
    labels than another client's.

    Each client is its own control -- its own-correlation minus the mean of its
    correlations against every other client's test profile. The pairing is what
    makes the number stable: comparing one group's median against another
    group's, using a single arbitrary mismatched pairing, resolves on the
    pairing rather than on the data (measured: sweeping the offset moved that
    statistic between -0.05 and +0.20 on FEMNIST).

    This is what a locally fitted head has to work with. Where it is near zero,
    the client's held-out labels look like anyone's, and there is nothing for a
    head to specialise to -- either because the client genuinely is not skewed,
    or because its test shard is too small to show that it is.
    """
    def profiles(labels, index_lists):
        labels = np.asarray(labels).reshape(-1)
        counts = np.stack(
            [np.bincount(labels[idx], minlength=num_classes).astype(float) for idx in index_lists]
        )
        return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)

    def unit_centred(matrix):
        centred = matrix - matrix.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centred, axis=1, keepdims=True)
        return centred / np.where(norms == 0, 1.0, norms)

    if len(shards) != len(test_shards):
        raise ValueError(
            f"shard lists disagree on client count: {len(shards)} vs {len(test_shards)}"
        )
    correlations = unit_centred(profiles(train_labels, shards)) @ unit_centred(
        profiles(test_labels, test_shards)
    ).T
    own = np.diag(correlations).copy()
    np.fill_diagonal(correlations, np.nan)
    return own - np.nanmean(correlations, axis=1)


@dataclass(frozen=True)
class WireSaving:
    """What withholding the head saves per model transfer."""

    spec_name: str
    parameters_total: int
    parameters_shared: int
    parameters_head: int
    payload_bytes_full: int
    payload_bytes_shared: int
    proto_bytes_full: int
    proto_bytes_shared: int

    @property
    def head_parameter_fraction(self) -> float:
        return self.parameters_head / self.parameters_total

    @property
    def proto_bytes_saved_fraction(self) -> float:
        return 1.0 - (self.proto_bytes_shared / self.proto_bytes_full)

    def to_dict(self) -> dict:
        return {
            "spec": self.spec_name,
            "parameters_total": self.parameters_total,
            "parameters_shared": self.parameters_shared,
            "parameters_head": self.parameters_head,
            "head_parameter_fraction": self.head_parameter_fraction,
            "payload_bytes_full": self.payload_bytes_full,
            "payload_bytes_shared": self.payload_bytes_shared,
            "proto_bytes_full": self.proto_bytes_full,
            "proto_bytes_shared": self.proto_bytes_shared,
            "proto_bytes_saved": self.proto_bytes_full - self.proto_bytes_shared,
            "proto_bytes_saved_fraction": self.proto_bytes_saved_fraction,
        }


def wire_saving(spec) -> WireSaving:
    """Measure, do not assume, what personalized mode removes from the wire.

    Both figures are reported because they differ: the float32 payload loses
    exactly four bytes per withheld parameter, while the protobuf-framed message
    also loses one tensor's name, shape and length prefix. The framed figure is
    the one the server's bytes-transferred metric counts.
    """
    from .serialization import proto_nbytes, shared_weights_to_proto, weights_to_proto

    weights = [np.zeros(s, dtype=np.float32) for s in spec.canonical_shapes()]
    shared, _head = spec.split_weights(weights)
    return WireSaving(
        spec_name=spec.name,
        parameters_total=spec.parameter_count(),
        parameters_shared=spec.shared_parameter_count(),
        parameters_head=spec.personal_parameter_count(),
        payload_bytes_full=int(sum(w.nbytes for w in weights)),
        payload_bytes_shared=int(sum(w.nbytes for w in shared)),
        proto_bytes_full=proto_nbytes(weights_to_proto(weights, names=spec.canonical_names())),
        proto_bytes_shared=proto_nbytes(shared_weights_to_proto(spec, weights)),
    )
