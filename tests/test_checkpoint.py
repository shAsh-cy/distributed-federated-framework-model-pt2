"""Checkpoint round-trips: the artifact a completed run leaves behind (audit M2)."""

from __future__ import annotations

import numpy as np
import pytest

from fl.checkpoint import CheckpointError, load_checkpoint, save_checkpoint


def _weights() -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    return [
        rng.normal(size=(3, 3, 1, 8)).astype(np.float32),
        rng.normal(size=(8,)).astype(np.float32),
        rng.normal(size=(128, 10)).astype(np.float32),
    ]


class TestRoundTrip:
    def test_weights_and_header_survive_exactly(self, tmp_path):
        weights = _weights()
        written = save_checkpoint(
            tmp_path / "model.npz",
            weights,
            model_name="small_cnn",
            config={"training": {"rounds": 20}},
            metadata={"rounds_completed": 20},
        )
        loaded, header = load_checkpoint(written)
        assert header["model"] == "small_cnn"
        assert header["num_tensors"] == 3
        assert header["config"]["training"]["rounds"] == 20
        assert header["metadata"]["rounds_completed"] == 20
        for original, restored in zip(weights, loaded, strict=True):
            np.testing.assert_array_equal(original, restored)
            assert restored.dtype == original.dtype

    def test_suffix_is_added_when_missing(self, tmp_path):
        written = save_checkpoint(tmp_path / "model", _weights(), model_name="small_cnn")
        assert written.suffix == ".npz"
        loaded, _ = load_checkpoint(written)
        assert len(loaded) == 3


class TestErrors:
    def test_empty_weights_rejected(self, tmp_path):
        with pytest.raises(CheckpointError, match="no tensors"):
            save_checkpoint(tmp_path / "x.npz", [], model_name="small_cnn")

    def test_missing_file_is_a_clear_error(self, tmp_path):
        with pytest.raises(CheckpointError, match="no checkpoint"):
            load_checkpoint(tmp_path / "absent.npz")

    def test_foreign_npz_rejected(self, tmp_path):
        path = tmp_path / "foreign.npz"
        np.savez(path, something=np.zeros(3))
        with pytest.raises(CheckpointError, match="no header"):
            load_checkpoint(path)
