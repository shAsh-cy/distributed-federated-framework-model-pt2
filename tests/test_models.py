"""Tests for the Keras model definitions."""

from __future__ import annotations

import numpy as np
import pytest

from fl.models import (
    INPUT_SHAPE,
    NUM_CLASSES,
    SMALL_CNN_PARAMS,
    build_model,
    build_small_cnn,
    count_parameters,
    weights_nbytes,
)


def test_parameter_count_is_exactly_as_documented():
    """The documented parameter count is load-bearing: it is the transfer size."""
    model = build_small_cnn(seed=0)
    assert count_parameters(model) == SMALL_CNN_PARAMS == 225_034


def test_parameter_count_matches_hand_computed_layer_breakdown():
    conv1 = (3 * 3 * 1) * 32 + 32
    conv2 = (3 * 3 * 32) * 64 + 64
    dense1 = (5 * 5 * 64) * 128 + 128
    logits = 128 * NUM_CLASSES + NUM_CLASSES
    assert conv1 + conv2 + dense1 + logits == SMALL_CNN_PARAMS


def test_transfer_size_is_under_one_mebibyte():
    """One model transfer must stay small enough for a 128 MB gRPC limit."""
    model = build_small_cnn(seed=0)
    nbytes = weights_nbytes(model.get_weights())
    assert nbytes == SMALL_CNN_PARAMS * 4 == 900_136
    assert nbytes < 1024 * 1024


def test_output_shape_is_logits_over_ten_classes():
    model = build_small_cnn(seed=0)
    out = model(np.zeros((7, *INPUT_SHAPE), dtype="float32"))
    assert tuple(out.shape) == (7, NUM_CLASSES)


def test_same_seed_gives_identical_initial_weights():
    """The server builds the initial global model from the seed alone."""
    a = build_small_cnn(seed=123).get_weights()
    b = build_small_cnn(seed=123).get_weights()
    for wa, wb in zip(a, b, strict=False):
        np.testing.assert_array_equal(wa, wb)


def test_different_seeds_give_different_initial_weights():
    a = build_small_cnn(seed=1).get_weights()
    b = build_small_cnn(seed=2).get_weights()
    assert any(not np.array_equal(wa, wb) for wa, wb in zip(a, b, strict=False))


def test_weights_round_trip_through_set_weights():
    model = build_small_cnn(seed=0)
    original = model.get_weights()
    perturbed = [w + 0.5 for w in original]
    model.set_weights(perturbed)
    for got, want in zip(model.get_weights(), perturbed, strict=False):
        np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_weight_list_structure_is_stable():
    """Aggregation zips weight lists positionally, so ordering must not drift."""
    weights = build_small_cnn(seed=0).get_weights()
    assert [w.shape for w in weights] == [
        (3, 3, 1, 32),
        (32,),
        (3, 3, 32, 64),
        (64,),
        (1600, 128),
        (128,),
        (128, 10),
        (10,),
    ]


def test_build_model_dispatches_by_name():
    assert build_model("small_cnn", seed=0).name == "small_cnn"


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("resnet50")


# -- femnist_cnn -------------------------------------------------------------


def test_femnist_parameter_count_is_exactly_as_documented():
    from fl.models import FEMNIST_CNN_PARAMS, build_femnist_cnn

    model = build_femnist_cnn(seed=0)
    assert count_parameters(model) == FEMNIST_CNN_PARAMS == 231_742


def test_femnist_parameter_count_matches_hand_computed_layer_breakdown():
    from fl.models import FEMNIST_CNN_PARAMS

    conv1 = (3 * 3 * 1) * 32 + 32
    conv2 = (3 * 3 * 32) * 64 + 64
    dense1 = 1600 * 128 + 128
    logits = 128 * 62 + 62
    assert conv1 + conv2 + dense1 + logits == FEMNIST_CNN_PARAMS


def test_femnist_output_is_logits_over_62_classes():
    from fl.models import build_femnist_cnn

    model = build_femnist_cnn(seed=0)
    out = model(np.zeros((2, 28, 28, 1), dtype=np.float32))
    assert out.shape == (2, 62)


def test_femnist_shares_backbone_with_small_cnn():
    """All layers except the logits layer have identical shapes across the two."""
    from fl.models import build_femnist_cnn, build_small_cnn

    small = build_small_cnn(seed=0).get_weights()
    fem = build_femnist_cnn(seed=0).get_weights()
    assert len(small) == len(fem)
    for a, b in zip(small[:-2], fem[:-2], strict=True):
        assert a.shape == b.shape
    assert fem[-2].shape == (128, 62) and fem[-1].shape == (62,)


def test_build_model_registry_knows_femnist():
    from fl.models import build_model

    assert build_model("femnist_cnn", seed=1).output_shape == (None, 62)
