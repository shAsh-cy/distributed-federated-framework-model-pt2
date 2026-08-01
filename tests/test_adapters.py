"""Adapter correctness — the tests this branch exists for.

Round trips assert *exact* equality, not tolerance: every conversion is a pure
axis permutation or rename, so any drift at all means a real bug. Forward
parity is asserted within float32 tolerance because the two frameworks order
their floating-point reductions differently.
"""

from __future__ import annotations

import numpy as np
import pytest

from fl.adapters import AdapterError, TFAdapter, TorchAdapter, make_adapter
from fl.aggregation import ClientUpdate, FedAvgAggregator
from fl.archspec import SMALL_CNN_SPEC, build_tf, build_torch
from tests.test_archspec import BN_SPEC

RNG = np.random.default_rng(3)


def _random_canonical(spec):
    return [RNG.standard_normal(s).astype(np.float32) for s in spec.canonical_shapes()]


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


class TestRoundTrips:
    def test_tf_to_canonical_to_torch_to_canonical_to_tf_is_exact(self):
        """The item-5 round trip: TF -> canonical -> torch -> canonical -> TF."""
        tf_model = build_tf(SMALL_CNN_SPEC, seed=11)
        tf_ad = TFAdapter(SMALL_CNN_SPEC)
        torch_ad = TorchAdapter(SMALL_CNN_SPEC)

        canonical_1 = tf_ad.to_canonical(tf_model)

        net = build_torch(SMALL_CNN_SPEC)
        torch_ad.from_canonical(net, canonical_1)
        canonical_2 = torch_ad.to_canonical(net)

        tf_model_2 = build_tf(SMALL_CNN_SPEC, seed=99)  # different init, then overwritten
        tf_ad.from_canonical(tf_model_2, canonical_2)
        final = tf_ad.to_canonical(tf_model_2)

        for name, a, b in zip(SMALL_CNN_SPEC.canonical_names(), canonical_1, final, strict=True):
            assert np.array_equal(a, b), f"round trip diverged at {name}"

    def test_batchnorm_round_trip_is_exact(self):
        canonical = _random_canonical(BN_SPEC)
        # Variance must be positive for the tensors to be plausible BN state.
        canonical[5] = np.abs(canonical[5]) + 0.1
        net = build_torch(BN_SPEC)
        ad = TorchAdapter(BN_SPEC)
        ad.from_canonical(net, canonical)
        back = ad.to_canonical(net)
        for name, a, b in zip(BN_SPEC.canonical_names(), canonical, back, strict=True):
            assert np.array_equal(a, b), f"BN round trip diverged at {name}"

    def test_batchnorm_statistics_map_to_the_right_slots(self):
        """gamma/beta/mean/var must land on weight/bias/running_mean/running_var."""
        canonical = _random_canonical(BN_SPEC)
        gamma = np.full(4, 2.0, np.float32)
        beta = np.full(4, -1.0, np.float32)
        mean = np.full(4, 0.25, np.float32)
        var = np.full(4, 9.0, np.float32)
        canonical[2:6] = [gamma, beta, mean, var]

        net = build_torch(BN_SPEC)
        TorchAdapter(BN_SPEC).from_canonical(net, canonical)
        bn = net.blocks.bn1
        assert np.allclose(bn.weight.detach().numpy(), 2.0)
        assert np.allclose(bn.bias.detach().numpy(), -1.0)
        assert np.allclose(bn.running_mean.numpy(), 0.25)
        assert np.allclose(bn.running_var.numpy(), 9.0)

    def test_shape_mismatch_is_rejected_by_both_adapters(self):
        wrong = _random_canonical(SMALL_CNN_SPEC)
        wrong[0] = wrong[0].T.copy()  # break the conv kernel layout
        with pytest.raises(AdapterError, match="do not match spec"):
            TorchAdapter(SMALL_CNN_SPEC).from_canonical(build_torch(SMALL_CNN_SPEC), wrong)
        with pytest.raises(AdapterError, match="do not match spec"):
            TFAdapter(SMALL_CNN_SPEC).from_canonical(build_tf(SMALL_CNN_SPEC, seed=0), wrong)

    def test_registry(self):
        assert isinstance(make_adapter("tensorflow", SMALL_CNN_SPEC), TFAdapter)
        assert isinstance(make_adapter("torch", SMALL_CNN_SPEC), TorchAdapter)
        with pytest.raises(ValueError, match="unknown framework"):
            make_adapter("jax", SMALL_CNN_SPEC)


# ---------------------------------------------------------------------------
# Forward parity
# ---------------------------------------------------------------------------


class TestForwardParity:
    def test_identical_weights_and_batch_give_matching_outputs(self):
        """The layout contract is only real if the *functions* agree."""
        import torch

        tf_model = build_tf(SMALL_CNN_SPEC, seed=5)
        canonical = TFAdapter(SMALL_CNN_SPEC).to_canonical(tf_model)
        net = build_torch(SMALL_CNN_SPEC)
        TorchAdapter(SMALL_CNN_SPEC).from_canonical(net, canonical)
        net.eval()

        batch = RNG.random((4, 28, 28, 1)).astype(np.float32)  # NHWC
        tf_out = tf_model(batch, training=False).numpy()
        torch_out = net(torch.from_numpy(batch.transpose(0, 3, 1, 2).copy())).detach().numpy()

        assert tf_out.shape == torch_out.shape == (4, 10)
        np.testing.assert_allclose(tf_out, torch_out, atol=1e-4, rtol=1e-4)

    def test_forward_parity_with_batchnorm(self):
        """BN inference must use the loaded moving statistics identically."""
        import torch

        tf_model = build_tf(BN_SPEC, seed=5)
        # Give BN non-trivial statistics so a mapping swap cannot cancel out.
        weights = tf_model.get_weights()
        weights[2] = np.linspace(0.5, 2.0, 4).astype(np.float32)  # gamma
        weights[3] = np.linspace(-1.0, 1.0, 4).astype(np.float32)  # beta
        weights[4] = np.linspace(-0.2, 0.2, 4).astype(np.float32)  # moving_mean
        weights[5] = np.linspace(0.5, 1.5, 4).astype(np.float32)  # moving_variance
        tf_model.set_weights(weights)

        canonical = TFAdapter(BN_SPEC).to_canonical(tf_model)
        net = build_torch(BN_SPEC)
        TorchAdapter(BN_SPEC).from_canonical(net, canonical)
        net.eval()

        batch = RNG.random((3, 8, 8, 1)).astype(np.float32)
        tf_out = tf_model(batch, training=False).numpy()
        torch_out = net(torch.from_numpy(batch.transpose(0, 3, 1, 2).copy())).detach().numpy()
        np.testing.assert_allclose(tf_out, torch_out, atol=1e-4, rtol=1e-4)

    def test_flatten_order_is_load_bearing(self):
        """Prove the NHWC-flatten permutation matters: skipping it breaks parity.

        Guards against the failure mode where both models are 'the same shape'
        and quietly compute different functions.
        """
        import torch

        tf_model = build_tf(SMALL_CNN_SPEC, seed=5)
        canonical = TFAdapter(SMALL_CNN_SPEC).to_canonical(tf_model)
        net = build_torch(SMALL_CNN_SPEC)
        TorchAdapter(SMALL_CNN_SPEC).from_canonical(net, canonical)
        net.eval()

        batch = RNG.random((4, 28, 28, 1)).astype(np.float32)
        x = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy())

        # Manually run the conv stack, then flatten NCHW-natively (the wrong way).
        with torch.no_grad():
            h = torch.relu(net.blocks.conv1(x))
            h = net.blocks.pool1(h)
            h = torch.relu(net.blocks.conv2(h))
            h = net.blocks.pool2(h)
            wrong = h.reshape(h.shape[0], -1)  # C*H*W order: no permute
            wrong = torch.relu(net.blocks.dense1(wrong))
            wrong = net.blocks.logits(wrong).numpy()

        tf_out = tf_model(batch, training=False).numpy()
        assert not np.allclose(tf_out, wrong, atol=1e-3), (
            "NCHW-order flatten unexpectedly matched; the parity test would be vacuous"
        )


# ---------------------------------------------------------------------------
# Framework-blind aggregation
# ---------------------------------------------------------------------------


class TestMixedPoolAggregation:
    def test_mixed_pool_equals_single_framework_pool_exactly(self):
        """Aggregation is numpy; identical numeric content must aggregate
        identically no matter which framework the bytes travelled through."""
        spec = SMALL_CNN_SPEC
        tf_ad, torch_ad = TFAdapter(spec), TorchAdapter(spec)
        initial = build_tf(spec, seed=1).get_weights()

        # Three clients' updates as canonical numeric content.
        contents = [_random_canonical(spec) for _ in range(3)]
        sizes = [10, 100, 1000]

        # Pool A: all three straight from canonical (single-framework path).
        pool_a = [
            ClientUpdate(f"a{i}", [w.copy() for w in c], n)
            for i, (c, n) in enumerate(zip(contents, sizes, strict=True))
        ]

        # Pool B: the same content, but clients 0 and 2 pushed theirs through a
        # torch module and back, client 1 through a keras model and back.
        via_torch = []
        for c in (contents[0], contents[2]):
            net = build_torch(spec)
            torch_ad.from_canonical(net, c)
            via_torch.append(torch_ad.to_canonical(net))
        model = build_tf(spec, seed=2)
        tf_ad.from_canonical(model, contents[1])
        via_tf = tf_ad.to_canonical(model)
        pool_b = [
            ClientUpdate("b0", via_torch[0], sizes[0]),
            ClientUpdate("b1", via_tf, sizes[1]),
            ClientUpdate("b2", via_torch[1], sizes[2]),
        ]

        agg = FedAvgAggregator()
        out_a = agg.aggregate(pool_a, [w.copy() for w in initial])
        out_b = agg.aggregate(pool_b, [w.copy() for w in initial])
        for name, a, b in zip(spec.canonical_names(), out_a, out_b, strict=True):
            assert np.array_equal(a, b), f"mixed pool diverged at {name}"
