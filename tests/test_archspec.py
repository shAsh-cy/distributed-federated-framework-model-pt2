"""Architecture spec: one definition, two frameworks, provably the same shape.

The parameter-count and per-layer-shape assertions here are item-level
guarantees: if either builder drifts from the spec — a transposed kernel, a
mis-sized flatten, a forgotten bias — these tests name the layer that moved.
"""

from __future__ import annotations

import numpy as np
import pytest

from fl.archspec import (
    FEMNIST_CNN_SPEC,
    SMALL_CNN_SPEC,
    ArchSpec,
    BatchNorm,
    Conv2D,
    Dense,
    Flatten,
    MaxPool2D,
    build_tf,
    build_torch,
)
from fl.models import FEMNIST_CNN_PARAMS, SMALL_CNN_PARAMS, build_small_cnn

#: A spec exercising BatchNorm, which the shipped architectures do not use but
#: the adapters must convert correctly.
BN_SPEC = ArchSpec(
    name="bn_probe",
    input_shape=(8, 8, 1),
    layers=(
        Conv2D(4, 3, "conv1"),
        BatchNorm("bn1"),
        MaxPool2D("pool1"),
        Flatten("flatten"),
        Dense(5, "logits"),
    ),
)


class TestSpecArithmetic:
    def test_small_cnn_spec_parameter_count_matches_recorded(self):
        assert SMALL_CNN_SPEC.parameter_count() == SMALL_CNN_PARAMS == 225_034

    def test_femnist_cnn_spec_parameter_count_matches_recorded(self):
        assert FEMNIST_CNN_SPEC.parameter_count() == FEMNIST_CNN_PARAMS == 231_742

    def test_canonical_shapes_are_the_documented_layout(self):
        shapes = SMALL_CNN_SPEC.canonical_shapes()
        assert shapes[0] == (3, 3, 1, 32)  # Conv2D (h, w, in, out)
        assert shapes[2] == (3, 3, 32, 64)
        assert shapes[4] == (1600, 128)  # Dense (in, out), 5*5*64 flatten
        assert shapes[6] == (128, 10)

    def test_canonical_names_follow_spec_order(self):
        names = SMALL_CNN_SPEC.canonical_names()
        assert names[:4] == ["conv1/kernel", "conv1/bias", "conv2/kernel", "conv2/bias"]
        assert names[-2:] == ["logits/kernel", "logits/bias"]

    def test_batchnorm_contributes_four_named_tensors(self):
        names = BN_SPEC.canonical_names()
        assert names[2:6] == ["bn1/gamma", "bn1/beta", "bn1/moving_mean", "bn1/moving_variance"]
        assert BN_SPEC.canonical_shapes()[2:6] == [(4,)] * 4

    def test_invalid_specs_rejected(self):
        with pytest.raises(ValueError, match="must follow a flatten"):
            ArchSpec("bad", (8, 8, 1), (Dense(3, "d"),))
        with pytest.raises(ValueError, match="unsupported activation"):
            ArchSpec("bad", (8, 8, 1), (Conv2D(4, 3, "c", activation="gelu"),))


class TestTfBuilder:
    def test_tf_model_matches_spec_shapes_and_count(self):
        model = build_tf(SMALL_CNN_SPEC, seed=0)
        got = [tuple(w.shape) for w in model.get_weights()]
        assert got == SMALL_CNN_SPEC.canonical_shapes()
        assert sum(int(np.prod(s)) for s in got) == SMALL_CNN_PARAMS

    def test_models_module_delegates_to_the_spec(self):
        """build_small_cnn and the spec builder are the same construction."""
        a = build_small_cnn(seed=7).get_weights()
        b = build_tf(SMALL_CNN_SPEC, seed=7).get_weights()
        for x, y in zip(a, b, strict=True):
            assert np.array_equal(x, y)

    def test_tf_batchnorm_weight_order_is_canonical(self):
        model = build_tf(BN_SPEC, seed=0)
        # keras get_weights order for BN: gamma, beta, moving_mean, moving_variance
        weights = model.get_weights()
        assert [tuple(w.shape) for w in weights] == BN_SPEC.canonical_shapes()
        gamma, beta, mean, var = weights[2:6]
        assert np.allclose(gamma, 1.0) and np.allclose(beta, 0.0)
        assert np.allclose(mean, 0.0) and np.allclose(var, 1.0)


class TestTorchBuilder:
    def test_torch_parameter_count_matches_spec(self):
        import torch

        net = build_torch(SMALL_CNN_SPEC)
        total = sum(p.numel() for p in net.parameters())
        assert total == SMALL_CNN_PARAMS
        assert isinstance(net, torch.nn.Module)

    def test_torch_femnist_parameter_count(self):
        net = build_torch(FEMNIST_CNN_SPEC)
        assert sum(p.numel() for p in net.parameters()) == FEMNIST_CNN_PARAMS

    def test_torch_batchnorm_counts_moving_statistics(self):
        net = build_torch(BN_SPEC)
        params = sum(p.numel() for p in net.parameters())
        buffers = sum(
            b.numel() for name, b in net.named_buffers() if "num_batches_tracked" not in name
        )
        assert params + buffers == BN_SPEC.parameter_count()

    def test_torch_forward_shape(self):
        import torch

        net = build_torch(SMALL_CNN_SPEC)
        out = net(torch.zeros(2, 1, 28, 28))
        assert tuple(out.shape) == (2, 10)

    def test_per_layer_shapes_agree_after_canonicalisation(self):
        """Torch native shapes, transposed per the documented layout, equal TF's."""
        net = build_torch(SMALL_CNN_SPEC)
        tf_shapes = SMALL_CNN_SPEC.canonical_shapes()
        state = dict(net.named_parameters())
        # conv kernels: torch (out, in, h, w) -> canonical (h, w, in, out)
        assert tuple(state["blocks.conv1.weight"].permute(2, 3, 1, 0).shape) == tf_shapes[0]
        assert tuple(state["blocks.conv2.weight"].permute(2, 3, 1, 0).shape) == tf_shapes[2]
        # dense kernels: torch (out, in) -> canonical (in, out)
        assert tuple(state["blocks.dense1.weight"].T.shape) == tf_shapes[4]
        assert tuple(state["blocks.logits.weight"].T.shape) == tf_shapes[6]


# ---------------------------------------------------------------------------
# The personal-layers marker
# ---------------------------------------------------------------------------


class TestPersonalLayers:
    """The backbone/head split: one definition, read by adapters, wire and harness."""

    def test_shipped_specs_mark_the_classifier_as_the_head(self):
        assert SMALL_CNN_SPEC.personal_layers == ("logits",)
        assert FEMNIST_CNN_SPEC.personal_layers == ("logits",)
        assert SMALL_CNN_SPEC.personal_names() == ["logits/kernel", "logits/bias"]
        assert SMALL_CNN_SPEC.shared_names()[-2:] == ["dense1/kernel", "dense1/bias"]

    def test_head_and_backbone_partition_the_parameter_count(self):
        for spec in (SMALL_CNN_SPEC, FEMNIST_CNN_SPEC):
            assert (
                spec.shared_parameter_count() + spec.personal_parameter_count()
                == spec.parameter_count()
            )
        # The two datasets' heads differ by an order of magnitude in share, so a
        # communication saving quoted for one is wrong for the other.
        assert FEMNIST_CNN_SPEC.personal_parameter_count() == 128 * 62 + 62
        assert SMALL_CNN_SPEC.personal_parameter_count() == 128 * 10 + 10
        assert SMALL_CNN_SPEC.shared_parameter_count() == FEMNIST_CNN_SPEC.shared_parameter_count()

    def test_mask_is_aligned_with_the_canonical_order(self):
        mask = FEMNIST_CNN_SPEC.personal_mask()
        names = FEMNIST_CNN_SPEC.canonical_names()
        assert len(mask) == len(names)
        head = [n for n, p in zip(names, mask, strict=True) if p]
        assert head == ["logits/kernel", "logits/bias"]
        assert mask == [False] * 6 + [True] * 2

    def test_split_and_merge_are_exact_inverses(self):
        weights = [
            np.arange(int(np.prod(s)), dtype=np.float32).reshape(s)
            for s in FEMNIST_CNN_SPEC.canonical_shapes()
        ]
        shared, head = FEMNIST_CNN_SPEC.split_weights(weights)
        assert [tuple(w.shape) for w in shared] == FEMNIST_CNN_SPEC.shared_shapes()
        assert [tuple(w.shape) for w in head] == FEMNIST_CNN_SPEC.personal_shapes()
        for a, b in zip(weights, FEMNIST_CNN_SPEC.merge_weights(shared, head), strict=True):
            assert np.array_equal(a, b)

    def test_a_spec_without_a_marker_is_all_backbone(self):
        """FedAvg is the same code path over an empty marker, not a second path."""
        plain = ArchSpec("plain", (8, 8, 1), BN_SPEC.layers)
        assert plain.personal_layers == ()
        assert plain.personal_names() == [] and plain.personal_parameter_count() == 0
        assert plain.shared_names() == plain.canonical_names()
        weights = [np.zeros(s, np.float32) for s in plain.canonical_shapes()]
        shared, head = plain.split_weights(weights)
        assert head == []
        assert len(shared) == len(weights)
        assert len(plain.merge_weights(shared, head)) == len(weights)

    def test_a_head_may_span_several_layers_if_they_are_the_top_ones(self):
        deep_head = ArchSpec(
            "deep_head",
            SMALL_CNN_SPEC.input_shape,
            SMALL_CNN_SPEC.layers,
            personal_layers=("dense1", "logits"),
        )
        assert deep_head.personal_parameter_count() == 204_928 + 1_290
        assert deep_head.shared_names() == [
            "conv1/kernel",
            "conv1/bias",
            "conv2/kernel",
            "conv2/bias",
        ]

    def test_a_head_taken_from_the_middle_is_rejected(self):
        """A 'head' below an aggregated layer is not a representation/head split
        of anything, so the spec refuses to describe one."""
        with pytest.raises(ValueError, match="trailing run"):
            ArchSpec("mid", (28, 28, 1), SMALL_CNN_SPEC.layers, personal_layers=("dense1",))
        with pytest.raises(ValueError, match="trailing run"):
            ArchSpec("gap", (28, 28, 1), SMALL_CNN_SPEC.layers, personal_layers=("conv2", "logits"))

    def test_a_weightless_layer_cannot_be_a_head(self):
        with pytest.raises(ValueError, match="not weight-bearing"):
            ArchSpec("pooly", (28, 28, 1), SMALL_CNN_SPEC.layers, personal_layers=("flatten",))
        with pytest.raises(ValueError, match="not weight-bearing"):
            ArchSpec("absent", (28, 28, 1), SMALL_CNN_SPEC.layers, personal_layers=("nope",))

    def test_duplicates_are_rejected_on_both_sides(self):
        with pytest.raises(ValueError, match="personal_layers contains duplicates"):
            ArchSpec(
                "dup", (28, 28, 1), SMALL_CNN_SPEC.layers, personal_layers=("logits", "logits")
            )
        with pytest.raises(ValueError, match="duplicate layer name"):
            ArchSpec("dupname", (8, 8, 1), (Conv2D(4, 3, "c"), Flatten("f"), Dense(2, "c")))

    def test_split_rejects_a_weight_list_of_the_wrong_arity(self):
        with pytest.raises(ValueError, match="canonical tensors"):
            FEMNIST_CNN_SPEC.split_weights([np.zeros((1,), np.float32)])

    def test_merge_rejects_a_head_of_the_wrong_width(self):
        """The failure this catches is the quiet one: a 10-class head merged
        into a 62-class model would load, train and score -- on ten classes."""
        weights = [np.zeros(s, np.float32) for s in FEMNIST_CNN_SPEC.canonical_shapes()]
        shared, _head = FEMNIST_CNN_SPEC.split_weights(weights)
        wrong = [np.zeros(s, np.float32) for s in SMALL_CNN_SPEC.personal_shapes()]
        with pytest.raises(ValueError, match="head weights do not match"):
            FEMNIST_CNN_SPEC.merge_weights(shared, wrong)
        with pytest.raises(ValueError, match="backbone weights do not match"):
            FEMNIST_CNN_SPEC.merge_weights(shared[:-1], _head)

    def test_batchnorm_in_a_head_carries_all_four_of_its_tensors(self):
        """A BatchNorm head is describable here even though the Keras trainer
        rejects it: the split is about tensors, and BN owns four, not two."""
        bn_head = ArchSpec(
            "bn_head",
            BN_SPEC.input_shape,
            BN_SPEC.layers,
            personal_layers=("logits",),
        )
        assert bn_head.personal_names() == ["logits/kernel", "logits/bias"]
        assert "bn1/moving_mean" in bn_head.shared_names()
