"""RPC communication.

Covers: the server starts and serves, a client registers, model weights survive a
real gRPC round trip bit-for-bit, oversized payloads are refused rather than
truncated, and malformed requests are rejected cleanly instead of crashing the
server.
"""

from __future__ import annotations

import time

import grpc
import numpy as np
import pytest

from fl.models import build_small_cnn
from fl.proto import fl_comm_pb2, fl_comm_pb2_grpc
from fl.serialization import (
    SerializationError,
    proto_to_weights,
    weights_fingerprint,
    weights_to_proto,
)
from tests.helpers import PB, FakeClient, ServerHarness, make_config

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Server lifecycle and registration
# ---------------------------------------------------------------------------


def test_server_starts_and_accepts_connections():
    with ServerHarness() as h:
        channel = grpc.insecure_channel(h.address)
        # result() raising TimeoutError IS this test's assertion: the channel
        # either reaches READY within 10s or the test fails (audit S1).
        ready = grpc.channel_ready_future(channel).result(timeout=10)
        assert ready is None  # future resolves with None exactly on readiness
        channel.close()


def test_client_registers_and_receives_identity_and_shard():
    with ServerHarness(make_config(data={"num_clients": 3})) as h:
        c = FakeClient(h.address)
        response = c.register()
        assert response.accepted
        assert response.client_id
        assert 0 <= response.shard_index < 3
        assert response.num_clients == 3
        c.close()


def test_registration_without_a_desired_id_still_gets_a_unique_one():
    """Docker replicas register with no id; the server must not collide them."""
    with ServerHarness(make_config(data={"num_clients": 4})) as h:
        clients = [FakeClient(h.address) for _ in range(4)]
        ids = set()
        for c in clients:
            r = c.register()
            assert r.accepted
            ids.add(r.client_id)
        assert len(ids) == 4
        for c in clients:
            c.close()


def test_server_stops_cleanly():
    h = ServerHarness()
    h.server.start()
    h.server.stop(grace=0)
    channel = grpc.insecure_channel(h.address)
    with pytest.raises(grpc.FutureTimeoutError):
        grpc.channel_ready_future(channel).result(timeout=2)
    channel.close()


# ---------------------------------------------------------------------------
# Weights survive the wire unchanged
# ---------------------------------------------------------------------------


def test_real_model_weights_round_trip_through_grpc_unchanged():
    """The full 225k-parameter model must arrive bit-identical."""
    weights = build_small_cnn(seed=1).get_weights()
    before = weights_fingerprint(weights)

    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 20.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config, initial=weights) as h:
        a = FakeClient(h.address)
        a.register(desired="a")
        b = FakeClient(h.address)
        b.register(desired="b")

        h.run_rounds_in_background()
        import time

        deadline = time.monotonic() + 10
        while h.server._round == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        response = None
        for candidate in (a, b):
            r = candidate.poll()
            if r.action == PB.ROUND_ACTION_TRAIN:
                response = r
                break
        assert response is not None

        received = proto_to_weights(response.weights)
        assert weights_fingerprint(received) == before
        for got, want in zip(received, weights, strict=False):
            np.testing.assert_array_equal(got, want)

        # And back up again, unchanged.
        reply = (
            a.submit(1, response.model_version, weights=received)
            if a.client_id in h.server._cohort
            else b.submit(1, response.model_version, weights=received)
        )
        assert reply.status == PB.UPDATE_STATUS_ACCEPTED
        stored = h.server._updates[next(iter(h.server._updates))]
        assert weights_fingerprint(stored.weights) == before

        h._thread.join(timeout=30)
        a.close()
        b.close()


@pytest.mark.parametrize("shape", [(1,), (7,), (2, 3), (3, 3, 1, 32), (1600, 128)])
def test_arbitrary_tensor_shapes_round_trip(shape):
    original = [np.random.default_rng(0).standard_normal(shape).astype(np.float32)]
    decoded = proto_to_weights(weights_to_proto(original))
    np.testing.assert_array_equal(decoded[0], original[0])
    assert decoded[0].shape == shape


def test_round_trip_preserves_extreme_float_values():
    original = [np.array([0.0, -0.0, 1e-38, 3.4e38, -3.4e38], dtype=np.float32)]
    decoded = proto_to_weights(weights_to_proto(original))
    np.testing.assert_array_equal(decoded[0], original[0])


def test_tensor_names_are_preserved():
    msg = weights_to_proto([np.zeros(2, np.float32)], names=["conv1/kernel"])
    assert msg.tensors[0].name == "conv1/kernel"


def test_name_count_mismatch_rejected():
    with pytest.raises(SerializationError, match="got 2 names for 1 tensors"):
        weights_to_proto([np.zeros(2, np.float32)], names=["a", "b"])


# ---------------------------------------------------------------------------
# Oversized payloads
# ---------------------------------------------------------------------------


def test_oversized_payload_is_refused_not_truncated():
    """A payload beyond the negotiated limit must fail loudly."""
    config = make_config(
        data={"num_clients": 2}, training={"client_fraction": 1.0}, server={"max_message_mb": 1}
    )
    with ServerHarness(config) as h:
        c = FakeClient(h.address)
        c.register(desired="c0")

        # ~4 MB, comfortably past the 1 MB limit.
        huge = [np.zeros((1_000_000,), dtype=np.float32)]
        with pytest.raises(grpc.RpcError) as excinfo:
            c.stub.SubmitUpdate(
                PB.SubmitUpdateRequest(
                    client_id=c.client_id,
                    round=1,
                    model_version=0,
                    weights=weights_to_proto(huge),
                    num_examples=10,
                )
            )
        assert excinfo.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        c.close()


def test_server_survives_an_oversized_payload_and_keeps_serving():
    """The limit must not take the server down with it."""
    config = make_config(
        data={"num_clients": 2}, training={"client_fraction": 1.0}, server={"max_message_mb": 1}
    )
    with ServerHarness(config) as h:
        c = FakeClient(h.address)
        c.register(desired="c0")
        with pytest.raises(grpc.RpcError):
            c.stub.SubmitUpdate(
                PB.SubmitUpdateRequest(
                    client_id=c.client_id,
                    round=1,
                    model_version=0,
                    weights=weights_to_proto([np.zeros((1_000_000,), np.float32)]),
                    num_examples=10,
                )
            )
        # Still alive.
        follow_up = FakeClient(h.address)
        assert follow_up.register(desired="c1").accepted
        follow_up.close()
        c.close()


# ---------------------------------------------------------------------------
# Malformed requests
# ---------------------------------------------------------------------------


def test_truncated_tensor_buffer_is_rejected_by_the_decoder():
    """A buffer that disagrees with its shape must not be reshaped into whatever fits."""
    msg = fl_comm_pb2.ModelWeights()
    tensor = msg.tensors.add()
    tensor.name = "t0"
    tensor.dtype = fl_comm_pb2.TENSOR_DTYPE_FLOAT32
    tensor.shape.extend([4, 4])  # claims 64 bytes
    tensor.data = b"\x00" * 32  # carries 32
    with pytest.raises(SerializationError, match="declares shape .* but carries 32 bytes"):
        proto_to_weights(msg)


def test_server_rejects_a_malformed_payload_without_crashing():
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 10.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        import time

        c = FakeClient(h.address)
        c.register(desired="c0")
        other = FakeClient(h.address)
        other.register(desired="c1")

        h.run_rounds_in_background()
        deadline = time.monotonic() + 10
        while h.server._round == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        bad = fl_comm_pb2.ModelWeights()
        tensor = bad.tensors.add()
        tensor.dtype = fl_comm_pb2.TENSOR_DTYPE_FLOAT32
        tensor.name = "t0"
        tensor.shape.extend([2, 3])
        tensor.data = b"\x01" * 7  # not 24 bytes

        target = next(iter(h.server._cohort))
        stub = c.stub if c.client_id == target else other.stub
        reply = stub.SubmitUpdate(
            PB.SubmitUpdateRequest(
                client_id=target, round=1, model_version=0, weights=bad, num_examples=10
            )
        )
        assert reply.status == PB.UPDATE_STATUS_REJECTED_INVALID_PAYLOAD
        assert "carries 7 bytes" in reply.detail

        # Server still responds afterwards.
        assert h.server.num_registered == 2
        h._thread.join(timeout=30)
        c.close()
        other.close()


def test_wrong_shaped_weights_are_rejected():
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 10.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        import time

        c = FakeClient(h.address)
        c.register(desired="c0")
        other = FakeClient(h.address)
        other.register(desired="c1")
        h.run_rounds_in_background()
        deadline = time.monotonic() + 10
        while h.server._round == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        target = next(iter(h.server._cohort))
        stub = c.stub if c.client_id == target else other.stub
        reply = stub.SubmitUpdate(
            PB.SubmitUpdateRequest(
                client_id=target,
                round=1,
                model_version=0,
                weights=weights_to_proto([np.zeros((9, 9), np.float32)]),
                num_examples=10,
            )
        )
        assert reply.status == PB.UPDATE_STATUS_REJECTED_INVALID_PAYLOAD
        assert "do not match global model" in reply.detail
        h._thread.join(timeout=30)
        c.close()
        other.close()


def test_non_positive_num_examples_rejected_over_the_wire():
    config = make_config(
        data={"num_clients": 2},
        training={"client_fraction": 1.0, "rounds": 1},
        server={"round_deadline_seconds": 10.0, "min_clients_per_round": 1},
    )
    with ServerHarness(config) as h:
        import time

        c = FakeClient(h.address)
        c.register(desired="c0")
        other = FakeClient(h.address)
        other.register(desired="c1")
        h.run_rounds_in_background()
        deadline = time.monotonic() + 10
        while h.server._round == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        target = next(iter(h.server._cohort))
        client = c if c.client_id == target else other
        reply = client.submit(1, 0, num_examples=0)
        assert reply.status == PB.UPDATE_STATUS_REJECTED_INVALID_PAYLOAD
        assert "num_examples must be positive" in reply.detail
        h._thread.join(timeout=30)
        c.close()
        other.close()


def test_unknown_client_polling_is_refused_with_not_found():
    with ServerHarness() as h:
        channel = grpc.insecure_channel(h.address)
        stub = fl_comm_pb2_grpc.FederatedLearningStub(channel)
        with pytest.raises(grpc.RpcError) as excinfo:
            stub.GetGlobalModel(PB.GetGlobalModelRequest(client_id="ghost"))
        assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND
        channel.close()


# -- schema versioning (V2) --------------------------------------------------


class TestSchemaVersioning:
    def test_v1_client_is_rejected_with_reason(self):
        with ServerHarness() as h:
            c = FakeClient(h.address)
            response = c.register(protocol_version=PB.PROTOCOL_VERSION_V1)
            assert not response.accepted
            assert "protocol version mismatch" in response.rejection_reason
            c.close()

    def test_unspecified_version_is_rejected(self):
        with ServerHarness() as h:
            c = FakeClient(h.address)
            assert not c.register(protocol_version=PB.PROTOCOL_VERSION_UNSPECIFIED).accepted
            c.close()

    def test_unsupported_version_does_not_affect_the_round(self):
        """A V1 client is refused while V2 clients complete the round normally."""
        config = make_config(
            data={"num_clients": 2},
            training={"client_fraction": 1.0, "rounds": 1},
            server={"round_deadline_seconds": 5.0, "min_clients_per_round": 2},
        )
        with ServerHarness(config) as h:
            a, b = FakeClient(h.address, fill=1.0), FakeClient(h.address, fill=3.0)
            assert a.register(desired="a").accepted
            assert b.register(desired="b").accepted
            # The V1 client is refused between the good registrations and the
            # round; nothing downstream may change for the others.
            bad = FakeClient(h.address)
            assert not bad.register(protocol_version=PB.PROTOCOL_VERSION_V1).accepted

            h.run_rounds_in_background()
            deadline = time.monotonic() + 10.0
            while h.server._round < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert a.submit(1, 0).status == PB.UPDATE_STATUS_ACCEPTED
            assert b.submit(1, 0).status == PB.UPDATE_STATUS_ACCEPTED
            h._thread.join(timeout=30)
            assert h.error is None
            # Both accepted updates aggregated: mean of fills 1.0 and 3.0.
            assert np.allclose(h.server.global_weights()[0], 2.0)
            a.close()
            b.close()
            bad.close()

    def test_dtype_travels_and_unsupported_dtype_rejected(self):
        msg = weights_to_proto([np.ones((2, 2), dtype=np.float32)])
        assert msg.tensors[0].dtype == PB.TENSOR_DTYPE_FLOAT32
        assert np.array_equal(proto_to_weights(msg)[0], np.ones((2, 2)))

        msg.tensors[0].dtype = PB.TENSOR_DTYPE_UNSPECIFIED
        with pytest.raises(SerializationError, match="unsupported dtype"):
            proto_to_weights(msg)
