"""Save and load the final global model, so a completed run leaves something usable.

Audit finding M2 (docs/audit_v0_2.md): every execution path trained models and
discarded them at run end — metrics JSON was the only artifact. This module is
the fix: a framework-neutral ``.npz`` checkpoint (numpy arrays in canonical
wire order plus a JSON header) that the TF path loads with
``build_model(...).set_weights`` and the torch path loads through the adapter,
with no TensorFlow import needed to read the file itself.

Format: ``tensor_000..tensor_NNN`` arrays in the model's canonical order —
the same order the wire protocol uses — plus a ``header`` JSON string with
the model name, tensor count, and whatever config/metadata the caller wants
preserved. ``allow_pickle`` stays False on both ends.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CHECKPOINT_VERSION = 1
DEFAULT_CHECKPOINT_DIR = Path("data/checkpoints")


class CheckpointError(RuntimeError):
    """Unreadable, mismatched, or structurally invalid checkpoint."""


def save_checkpoint(
    path: str | Path,
    weights: list[np.ndarray],
    model_name: str,
    config: dict | None = None,
    metadata: dict | None = None,
) -> Path:
    """Write ``weights`` (canonical order) and a JSON header to ``path``.

    Returns the actual path written (numpy appends ``.npz`` if missing).
    """
    if not weights:
        raise CheckpointError("refusing to save a checkpoint with no tensors")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model_name,
        "num_tensors": len(weights),
        "config": config or {},
        "metadata": metadata or {},
    }
    arrays = {f"tensor_{i:03d}": np.asarray(w) for i, w in enumerate(weights)}
    np.savez(path, header=np.array(json.dumps(header)), **arrays)
    return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")


def load_checkpoint(path: str | Path) -> tuple[list[np.ndarray], dict]:
    """Read a checkpoint back: (weights in canonical order, header dict)."""
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"no checkpoint at {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "header" not in archive:
            raise CheckpointError(f"{path} has no header; not a checkpoint from this repo")
        header = json.loads(str(archive["header"]))
        expected = header.get("num_tensors")
        weights = []
        for i in range(expected):
            key = f"tensor_{i:03d}"
            if key not in archive:
                raise CheckpointError(f"{path} is missing {key} of {expected}")
            weights.append(archive[key])
    return weights, header
