"""Personalization: the state, the wire, and the evaluation isolation.

Three things must hold before a personalization number means anything, and each
gets a test that fails loudly rather than a comment saying it holds:

1. **The head is per-client state and stays that way.** No client's head is
   reachable from another client's training, and the store cannot be mutated
   through a reference it handed out.
2. **The head never reaches the wire.** Personalized encoding carries the
   backbone alone, and decoding *rejects* a payload carrying a head rather than
   dropping it -- silence there would leak the exact parameters personalization
   exists to keep local, and would leak them invisibly.
3. **Per-client evaluation pairs each client's head with that client's own
   held-out data, and no test sample reaches any training path.** The existing
   shard-leakage tests (tests/test_data.py, tests/test_femnist.py) cover the
   training half; these extend the same discipline to heads and to per-client
   test shards.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/ holds the harness the evaluation-isolation tests drive; it is not a
# package, and the batch scripts keep their heavy imports inside functions
# precisely so this import stays cheap.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fl.archspec import (
    FEMNIST_CNN_SPEC,
    SMALL_CNN_SPEC,
    ArchSpec,
    Conv2D,
    Dense,
    Flatten,
    MaxPool2D,
)
from fl.personalization import (
    HeadStore,
    PersonalizationError,
    distribution_summary,
    paired_delta_summary,
    weighted_mean,
    wire_saving,
)

RNG = np.random.default_rng(17)


def _head(spec, fill: float = 0.0) -> list[np.ndarray]:
    return [np.full(s, fill, dtype=np.float32) for s in spec.personal_shapes()]


def _full(spec) -> list[np.ndarray]:
    return [RNG.standard_normal(s).astype(np.float32) for s in spec.canonical_shapes()]


# ---------------------------------------------------------------------------
# 1. Per-client head state
# ---------------------------------------------------------------------------


class TestHeadStore:
    def test_untrained_client_reads_the_initial_head(self):
        store = HeadStore(FEMNIST_CNN_SPEC, _head(FEMNIST_CNN_SPEC, 0.25))
        assert store.updates("c0") == 0
        assert not store.has_trained("c0")
        assert all(np.allclose(w, 0.25) for w in store.get("c0"))

    def test_heads_are_isolated_between_clients(self):
        store = HeadStore(FEMNIST_CNN_SPEC, _head(FEMNIST_CNN_SPEC))
        store.put("a", _head(FEMNIST_CNN_SPEC, 1.0))
        store.put("b", _head(FEMNIST_CNN_SPEC, 2.0))
        assert all(np.allclose(w, 1.0) for w in store.get("a"))
        assert all(np.allclose(w, 2.0) for w in store.get("b"))
        assert all(np.allclose(w, 0.0) for w in store.get("c"))  # never trained

    def test_a_returned_head_cannot_be_used_to_mutate_the_store(self):
        """The copy-out is load-bearing, not defensive style.

        Local training mutates the array it was handed. If ``get`` returned the
        stored array, one client's round would rewrite the stored head of
        whichever client it was reading -- a cross-client write that no accuracy
        figure would reveal.
        """
        store = HeadStore(FEMNIST_CNN_SPEC, _head(FEMNIST_CNN_SPEC))
        store.put("a", _head(FEMNIST_CNN_SPEC, 1.0))
        borrowed = store.get("a")
        borrowed[0][:] = 99.0
        assert all(np.allclose(w, 1.0) for w in store.get("a"))

    def test_a_stored_head_cannot_be_mutated_through_the_caller_s_array(self):
        store = HeadStore(FEMNIST_CNN_SPEC, _head(FEMNIST_CNN_SPEC))
        mine = _head(FEMNIST_CNN_SPEC, 1.0)
        store.put("a", mine)
        mine[0][:] = 99.0
        assert all(np.allclose(w, 1.0) for w in store.get("a"))

    def test_update_counts_track_participation(self):
        store = HeadStore(FEMNIST_CNN_SPEC, _head(FEMNIST_CNN_SPEC))
        for _ in range(3):
            store.put("a", _head(FEMNIST_CNN_SPEC, 1.0))
        store.put("b", _head(FEMNIST_CNN_SPEC, 1.0))
        assert store.updates("a") == 3
        assert store.updates("b") == 1
        assert store.updates("c") == 0
        assert store.num_trained == 2
        assert store.participation(["a", "b", "c"]) == {"a": 3, "b": 1, "c": 0}

    def test_client_ids_are_normalised_so_int_and_str_are_one_client(self):
        """The harness indexes clients by int, the wire by string. Both must
        reach the same head, or a client would silently personalize twice."""
        store = HeadStore(FEMNIST_CNN_SPEC, _head(FEMNIST_CNN_SPEC))
        store.put(7, _head(FEMNIST_CNN_SPEC, 1.0))
        assert store.updates("7") == 1
        assert all(np.allclose(w, 1.0) for w in store.get("7"))

    def test_wrong_shaped_head_is_rejected(self):
        store = HeadStore(FEMNIST_CNN_SPEC, _head(FEMNIST_CNN_SPEC))
        with pytest.raises(PersonalizationError, match="does not match the head"):
            store.put("a", _head(SMALL_CNN_SPEC, 1.0))  # 10 classes, not 62
        with pytest.raises(PersonalizationError, match="initial head"):
            HeadStore(FEMNIST_CNN_SPEC, _head(SMALL_CNN_SPEC))


# ---------------------------------------------------------------------------
# 2. The wire
# ---------------------------------------------------------------------------


class TestPersonalizedWire:
    def test_personalized_encoding_carries_the_backbone_only(self):
        from fl.serialization import shared_weights_to_proto

        weights = _full(FEMNIST_CNN_SPEC)
        msg = shared_weights_to_proto(FEMNIST_CNN_SPEC, weights)
        names = [t.name for t in msg.tensors]
        assert names == FEMNIST_CNN_SPEC.shared_names()
        assert not set(names) & set(FEMNIST_CNN_SPEC.personal_names())
        assert len(msg.tensors) == len(weights) - 2  # logits kernel and bias withheld

    def test_backbone_round_trips_exactly(self):
        from fl.serialization import proto_to_shared_weights, shared_weights_to_proto

        weights = _full(FEMNIST_CNN_SPEC)
        shared, _head_part = FEMNIST_CNN_SPEC.split_weights(weights)
        back = proto_to_shared_weights(
            FEMNIST_CNN_SPEC, shared_weights_to_proto(FEMNIST_CNN_SPEC, weights)
        )
        for a, b in zip(shared, back, strict=True):
            assert np.array_equal(a, b)

    def test_recombining_with_the_local_head_restores_the_full_model(self):
        from fl.serialization import proto_to_personalized_weights, shared_weights_to_proto

        weights = _full(FEMNIST_CNN_SPEC)
        _shared, head = FEMNIST_CNN_SPEC.split_weights(weights)
        restored = proto_to_personalized_weights(
            FEMNIST_CNN_SPEC, shared_weights_to_proto(FEMNIST_CNN_SPEC, weights), head
        )
        for a, b in zip(weights, restored, strict=True):
            assert np.array_equal(a, b)

    def test_a_payload_carrying_a_head_tensor_is_refused(self):
        """Rejected, not silently stripped. A peer sending its head is either
        broken or leaking, and dropping the tensor would hide both."""
        from fl.serialization import SerializationError, proto_to_shared_weights, weights_to_proto

        weights = _full(FEMNIST_CNN_SPEC)
        full_msg = weights_to_proto(weights, names=FEMNIST_CNN_SPEC.canonical_names())
        with pytest.raises(SerializationError, match="must never reach the wire"):
            proto_to_shared_weights(FEMNIST_CNN_SPEC, full_msg)

    def test_a_payload_for_another_architecture_is_refused(self):
        from fl.serialization import SerializationError, proto_to_shared_weights, weights_to_proto

        shared, _head = SMALL_CNN_SPEC.split_weights(_full(SMALL_CNN_SPEC))
        msg = weights_to_proto(shared, names=[f"t{i}" for i in range(len(shared))])
        with pytest.raises(SerializationError, match="do not match the backbone"):
            proto_to_shared_weights(SMALL_CNN_SPEC, msg)

    def test_a_truncated_backbone_is_refused(self):
        from fl.serialization import SerializationError, proto_to_shared_weights, weights_to_proto

        shared, _head = FEMNIST_CNN_SPEC.split_weights(_full(FEMNIST_CNN_SPEC))
        msg = weights_to_proto(shared[:-1], names=FEMNIST_CNN_SPEC.shared_names()[:-1])
        with pytest.raises(SerializationError, match="do not match the backbone"):
            proto_to_shared_weights(FEMNIST_CNN_SPEC, msg)

    def test_saving_is_exactly_the_head_and_no_more(self):
        saving = wire_saving(FEMNIST_CNN_SPEC)
        assert saving.parameters_total == 231_742
        assert saving.parameters_head == 128 * 62 + 62 == 7_998
        assert saving.parameters_shared == 231_742 - 7_998
        assert saving.payload_bytes_full - saving.payload_bytes_shared == 7_998 * 4
        # Framed bytes drop by at least the payload, and by more: a whole
        # tensor's name, shape and length prefix go with it.
        assert saving.proto_bytes_full - saving.proto_bytes_shared > 7_998 * 4
        assert 0.03 < saving.head_parameter_fraction < 0.04
        assert saving.to_dict()["spec"] == "femnist_cnn"

    def test_the_small_cnn_head_is_a_much_smaller_share(self):
        """Stated so the write-up cannot quote one dataset's saving for both."""
        saving = wire_saving(SMALL_CNN_SPEC)
        assert saving.parameters_head == 128 * 10 + 10 == 1_290
        assert saving.head_parameter_fraction < 0.006


# ---------------------------------------------------------------------------
# 3. Reporting
# ---------------------------------------------------------------------------


class TestDistributionReporting:
    def test_summary_reports_the_tails_not_just_the_middle(self):
        values = list(np.linspace(0.0, 1.0, 101))
        s = distribution_summary(values)
        assert s["n"] == 101
        assert s["median"] == pytest.approx(0.5)
        assert s["decile_size"] == 10
        assert s["worst_decile_mean"] == pytest.approx(np.mean(np.linspace(0, 1, 101)[:10]))
        assert s["best_decile_mean"] == pytest.approx(np.mean(np.linspace(0, 1, 101)[-10:]))

    def test_worst_decile_is_a_mean_over_the_tail_not_a_percentile(self):
        """One catastrophic client must move the tail statistic. A p10 would
        not notice it; the mean of the worst decile does."""
        base = [0.8] * 20
        s_flat = distribution_summary(base)
        s_one_bad = distribution_summary([0.0] + base[1:])
        assert s_flat["worst_decile_mean"] == pytest.approx(0.8)
        assert s_one_bad["worst_decile_mean"] < 0.5

    def test_a_single_client_still_summarises(self):
        s = distribution_summary([0.4])
        assert s["n"] == 1 and s["decile_size"] == 1
        assert s["worst_decile_mean"] == s["best_decile_mean"] == pytest.approx(0.4)

    def test_empty_input_is_an_error_not_a_nan(self):
        with pytest.raises(ValueError, match="at least one value"):
            distribution_summary([])

    def test_paired_delta_counts_clients_not_populations(self):
        s = paired_delta_summary([0.5, 0.5, 0.5, 0.5], [0.9, 0.6, 0.5, 0.1])
        assert s["fraction_improved"] == pytest.approx(0.5)
        assert s["fraction_unchanged"] == pytest.approx(0.25)
        assert s["fraction_worsened"] == pytest.approx(0.25)
        assert s["mean"] == pytest.approx(0.025)
        assert s["min"] == pytest.approx(-0.4)

    def test_paired_delta_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="matching client counts"):
            paired_delta_summary([0.1, 0.2], [0.1])

    def test_weighted_and_unweighted_means_separate_under_skew(self):
        """The distinction the reporting insists on: 'accuracy over the pooled
        test set' and 'accuracy for the average client' are different numbers."""
        values, sizes = [1.0, 0.0], [1, 99]
        assert weighted_mean(values, sizes) == pytest.approx(0.01)
        assert distribution_summary(values)["mean"] == pytest.approx(0.5)

    def test_weighted_mean_rejects_degenerate_weights(self):
        with pytest.raises(ValueError, match="positive"):
            weighted_mean([0.5, 0.5], [0, 0])


# ---------------------------------------------------------------------------
# 4. Per-client evaluation isolation
# ---------------------------------------------------------------------------

#: A three-class spec small enough to build and evaluate in milliseconds; the
#: evaluation-pairing tests are about which head meets which shard, and a
#: 231,742-parameter model would only make them slower.
TINY_SPEC = ArchSpec(
    name="tiny_personal",
    input_shape=(8, 8, 1),
    layers=(
        Conv2D(2, 3, "conv1"),
        MaxPool2D("pool1"),
        Flatten("flatten"),
        Dense(4, "dense1", activation="relu"),
        Dense(3, "logits"),
    ),
    personal_layers=("logits",),
)


def _constant_predictor_head(spec, cls: int) -> list[np.ndarray]:
    """A head that predicts ``cls`` for every input, whatever the backbone says.

    Zero kernel, one-hot bias: the logits are the bias. That makes the measured
    accuracy a pure function of *which head met which labels*, so a mispairing
    shows up as a wrong number rather than as a slightly worse one.
    """
    kernel, bias = (np.zeros(s, dtype=np.float32) for s in spec.personal_shapes())
    bias[cls] = 1.0
    return [kernel, bias]


@pytest.fixture(scope="module")
def tiny_eval_setup():
    """A model, a test split, and per-client test shards with known labels."""
    from fl.archspec import build_tf
    from fl.data import Dataset

    rng = np.random.default_rng(5)
    # Client k's shard is mostly class k, with a known contaminating minority,
    # so "scored on its own shard" and "scored on the pooled set" differ.
    labels, shards, offset = [], [], 0
    for k in range(3):
        own = [k] * 8 + [(k + 1) % 3] * 2
        labels.extend(own)
        shards.append(np.arange(offset, offset + len(own), dtype=np.int64))
        offset += len(own)
    y = np.asarray(labels, dtype=np.int64)
    x = rng.random((len(y), 8, 8, 1)).astype(np.float32)
    return build_tf(TINY_SPEC, seed=0), Dataset(x=x, y=y), shards


class TestPerClientEvaluationIsolation:
    def test_each_client_is_scored_with_its_own_head_on_its_own_shard(self, tiny_eval_setup):
        """The pairing test.

        Client ``k`` gets a head that always answers ``k``; its shard is 80 %
        class ``k``. So every client must score exactly 0.8. Any mispairing --
        head of ``k`` against shard of ``j``, or the right head against the
        pooled set -- lands on 0.1 or 0.333, not 0.8.
        """
        import personalization_experiments as px

        model, test, test_shards = tiny_eval_setup
        store = HeadStore(TINY_SPEC, _constant_predictor_head(TINY_SPEC, 0))
        for k in range(3):
            store.put(k, _constant_predictor_head(TINY_SPEC, k))
        shared, _head = TINY_SPEC.split_weights(model.get_weights())

        rows = px.evaluate_per_client(
            model,
            TINY_SPEC,
            test=test,
            test_shards=test_shards,
            shared_weights=shared,
            head_store=store,
        )
        assert [r["client"] for r in rows] == [0, 1, 2]
        assert [r["accuracy"] for r in rows] == [pytest.approx(0.8)] * 3
        assert [r["test_samples"] for r in rows] == [10, 10, 10]
        assert [r["head_updates"] for r in rows] == [1, 1, 1]

    def test_the_global_arm_scores_one_model_on_every_client_shard(self, tiny_eval_setup):
        """The same shards under ``full_weights``: one model, so the spread
        across clients is the heterogeneity, not the pairing."""
        import personalization_experiments as px

        model, test, test_shards = tiny_eval_setup
        full = TINY_SPEC.merge_weights(
            TINY_SPEC.split_weights(model.get_weights())[0],
            _constant_predictor_head(TINY_SPEC, 1),
        )
        rows = px.evaluate_per_client(
            model, TINY_SPEC, test=test, test_shards=test_shards, full_weights=full
        )
        # A model that always answers 1: client 0 has two 1s, client 1 has
        # eight, client 2 has none.
        assert [r["accuracy"] for r in rows] == [
            pytest.approx(0.2),
            pytest.approx(0.8),
            pytest.approx(0.0),
        ]

    def test_a_client_with_no_held_out_data_is_reported_not_dropped(self, tiny_eval_setup):
        import personalization_experiments as px

        model, test, test_shards = tiny_eval_setup
        shards = [*test_shards, np.empty(0, dtype=np.int64)]
        rows = px.evaluate_per_client(
            model, TINY_SPEC, test=test, test_shards=shards, full_weights=model.get_weights()
        )
        assert len(rows) == 4
        assert rows[-1]["accuracy"] is None and rows[-1]["test_samples"] == 0

    def test_the_two_evaluation_modes_are_mutually_exclusive(self, tiny_eval_setup):
        import personalization_experiments as px

        model, test, test_shards = tiny_eval_setup
        with pytest.raises(ValueError, match="exactly one of"):
            px.evaluate_per_client(model, TINY_SPEC, test=test, test_shards=test_shards)
        with pytest.raises(ValueError, match="needs a head_store"):
            px.evaluate_per_client(
                model,
                TINY_SPEC,
                test=test,
                test_shards=test_shards,
                shared_weights=TINY_SPEC.split_weights(model.get_weights())[0],
            )


# ---------------------------------------------------------------------------
# 5. The training path never sees a test sample, and no head sees foreign data
# ---------------------------------------------------------------------------


def _row_digests(x: np.ndarray) -> set[bytes]:
    return {hashlib.sha1(np.ascontiguousarray(row).tobytes()).digest() for row in x}


@pytest.fixture(scope="module")
def synthetic_femnist(tmp_path_factory):
    """A six-writer FEMNIST cache in the real packed format.

    Random pixels, so byte-identical collisions between the splits are
    impossible and a content-level leak check is exact here -- unlike on the
    real cache, where LEAF's own 0.84 % duplicate rate sets the floor."""
    from fl.data import pack_femnist

    rng = np.random.default_rng(3)
    train, test, ids = [], [], []
    for i in range(6):
        n_train, n_test = 12 + 3 * i, 6
        classes = tuple(range(i, i + 3))
        train.append(
            (
                rng.random((n_train, 28, 28)).astype(np.float32),
                rng.choice(classes, size=n_train),
            )
        )
        test.append(
            (rng.random((n_test, 28, 28)).astype(np.float32), rng.choice(classes, size=n_test))
        )
        ids.append(f"writer_{i}")
    path = tmp_path_factory.mktemp("femnist_personal") / "cache.npz"
    np.savez_compressed(path, **pack_femnist(train, test, ids))
    return str(path)


@pytest.mark.slow
class TestTrainingIsolation:
    """A real (tiny) FedRep run, instrumented at the two points that matter."""

    @staticmethod
    def _run(monkeypatch, cache):
        import personalization_experiments as px
        from fl.data import load_femnist_per_client

        train, test, shards, test_shards = load_femnist_per_client(cache_path=cache)

        seen: list[tuple[bytes, bytes]] = []
        stored: list[tuple[str, int]] = []

        real_fit = px.StagedTrainer.fit

        def recording_fit(self, x, y, stages, batch_size, rng, shuffle=True):
            seen.append(
                (
                    hashlib.sha1(np.ascontiguousarray(x).tobytes()).digest(),
                    hashlib.sha1(np.ascontiguousarray(y).tobytes()).digest(),
                )
            )
            return real_fit(self, x, y, stages, batch_size, rng, shuffle)

        real_put = px.HeadStore.put

        def recording_put(self, client_id, head):
            # Pair this head with the fit that produced it: simulate() fits and
            # then immediately stores, so the last recorded fit is this client's.
            stored.append((str(client_id), len(seen) - 1))
            return real_put(self, client_id, head)

        monkeypatch.setattr(px.StagedTrainer, "fit", recording_fit)
        monkeypatch.setattr(px.HeadStore, "put", recording_put)

        result = px.simulate(
            model_name="femnist_cnn",
            train=train,
            test=test,
            shards=shards,
            test_shards=test_shards,
            method="fedrep",
            clients_per_round=3,
            rounds=2,
            local_epochs=2,
            head_epochs=1,
            batch_size=8,
            seed=1,
            label="isolation",
        )
        return result, train, test, shards, seen, stored

    def test_every_stored_head_was_trained_on_that_client_s_data_alone(
        self, monkeypatch, synthetic_femnist
    ):
        _result, train, _test, shards, seen, stored = self._run(monkeypatch, synthetic_femnist)
        assert stored, "no head was ever stored; the run did not exercise FedRep"
        for client_id, fit_index in stored:
            expected = hashlib.sha1(
                np.ascontiguousarray(train.x[shards[int(client_id)]]).tobytes()
            ).digest()
            assert seen[fit_index][0] == expected, (
                f"head for client {client_id} was fitted on data that is not client "
                f"{client_id}'s shard"
            )

    def test_no_test_sample_reaches_any_training_call(self, monkeypatch, synthetic_femnist):
        """Content-level, not index-level.

        Every array handed to local training is hashed and matched against the
        set of *training* shard digests. A test sample entering the fit -- by a
        mixed-up shard list, a pooled array, or a fine-tuning control reading the
        wrong split -- changes the digest and fails here.
        """
        _result, train, test, shards, seen, _stored = self._run(monkeypatch, synthetic_femnist)
        legitimate = {
            hashlib.sha1(np.ascontiguousarray(train.x[s]).tobytes()).digest() for s in shards
        }
        assert seen, "no training call was recorded"
        for x_digest, _y_digest in seen:
            assert x_digest in legitimate

        # And the stronger form: no individual held-out image appears in any
        # shard that training could have been handed.
        test_rows = _row_digests(test.x)
        for shard in shards:
            assert not (_row_digests(train.x[shard]) & test_rows)

    def test_a_never_sampled_client_is_reported_as_never_sampled(
        self, monkeypatch, synthetic_femnist
    ):
        """Six writers, three per round, two rounds: some client may never be
        sampled, and its 'personalized' accuracy is really the cold head. The
        record must say which clients those are rather than average them in
        silently."""
        result, _train, _test, _shards, _seen, _stored = self._run(monkeypatch, synthetic_femnist)
        participation = result["head_participation"]
        assert set(participation) == {str(i) for i in range(6)}
        assert sum(participation.values()) == 3 * 2  # cohort size x rounds
        never = [cid for cid, count in participation.items() if count == 0]
        assert result["summary"]["clients_never_sampled"] == len(never)
        assert all(row["head_updates"] == participation[str(row["client"])]
                   for row in result["per_client"])


@pytest.mark.slow
class TestArmsAreMatched:
    def test_fedavg_and_fedrep_run_the_same_loop_over_the_same_cohorts(
        self, monkeypatch, synthetic_femnist
    ):
        """Same seed, same population: the cohort sequence must be identical, so
        a difference between the arms is the algorithm and not the sampling."""
        import personalization_experiments as px
        from fl.data import load_femnist_per_client

        train, test, shards, test_shards = load_femnist_per_client(cache_path=synthetic_femnist)
        cohorts: dict[str, list] = {}
        # Captured before any patching: taking it inside run() would make the
        # second arm's recorder wrap the first arm's, and both would record.
        pristine_fit = px.StagedTrainer.fit

        def run(method):
            seen: list[bytes] = []

            def recording_fit(self, x, y, stages, batch_size, rng, shuffle=True):
                seen.append(hashlib.sha1(np.ascontiguousarray(x).tobytes()).digest())
                return pristine_fit(self, x, y, stages, batch_size, rng, shuffle)

            monkeypatch.setattr(px.StagedTrainer, "fit", recording_fit)
            result = px.simulate(
                model_name="femnist_cnn",
                train=train,
                test=test,
                shards=shards,
                test_shards=test_shards,
                method=method,
                clients_per_round=3,
                rounds=2,
                local_epochs=2,
                head_epochs=1,
                batch_size=8,
                seed=1,
                finetune_control=False,
                label=f"matched/{method}",
            )
            cohorts[method] = seen
            return result

        fedavg, fedrep = run("fedavg"), run("fedrep")
        assert cohorts["fedavg"] == cohorts["fedrep"]
        assert fedavg["local_epochs"] == fedrep["local_epochs"] == 2
        assert fedrep["head_epochs"] == 1 and fedrep["backbone_epochs"] == 1
        # FedAvg has a global model to evaluate; FedRep does not, and says so
        # rather than reporting a number assembled from an arbitrary head.
        assert fedavg["final_pooled_accuracy"] is not None
        assert fedrep["final_pooled_accuracy"] is None

    def test_fedrep_requires_both_stages_to_be_non_empty(self, synthetic_femnist):
        import personalization_experiments as px
        from fl.data import load_femnist_per_client

        train, test, shards, test_shards = load_femnist_per_client(cache_path=synthetic_femnist)
        for head_epochs in (0, 2):
            with pytest.raises(ValueError, match="head_epochs must be in"):
                px.simulate(
                    model_name="femnist_cnn",
                    train=train,
                    test=test,
                    shards=shards,
                    test_shards=test_shards,
                    method="fedrep",
                    clients_per_round=2,
                    rounds=1,
                    local_epochs=2,
                    head_epochs=head_epochs,
                    seed=1,
                )
