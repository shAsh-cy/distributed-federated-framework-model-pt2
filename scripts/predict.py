"""Load a saved checkpoint and classify test samples — proof the artifact works.

The other half of audit finding M2 (docs/audit_v0_2.md): saving a model is
only useful if someone can load it and get sane predictions back. This
script does exactly that, on either or both frameworks:

    python scripts/predict.py --checkpoint data/checkpoints/<run>.npz
    python scripts/predict.py --checkpoint results/model.npz --framework torch
    python scripts/predict.py --checkpoint results/model.npz --framework both

``both`` additionally checks cross-framework agreement: the same checkpoint
loaded into the TF model and through the torch adapter must produce the same
argmax on every sampled input (the repo's parity guarantee, applied to a
real artifact). torch is imported before TensorFlow, per the repo-wide rule.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fl.checkpoint import load_checkpoint  # noqa: E402

LOGGER = logging.getLogger("predict")

_DATASET_FOR_MODEL = {"small_cnn": "fashion_mnist", "femnist_cnn": "femnist"}


def _test_samples(model_name: str, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from fl.data import load_femnist

    if model_name == "femnist_cnn":
        _, test, _ = load_femnist()
        x, y = test.x, test.y
    else:
        from fl.data import load_fashion_mnist

        _, test = load_fashion_mnist()
        x, y = test.x, test.y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(count, len(x)), replace=False)
    return x[idx], y[idx]


def predict_tf(weights: list[np.ndarray], model_name: str, x: np.ndarray) -> np.ndarray:
    from fl.models import build_model

    model = build_model(model_name)
    model.set_weights(weights)
    return np.asarray(model.predict(x, verbose=0))


def predict_torch(weights: list[np.ndarray], model_name: str, x: np.ndarray) -> np.ndarray:
    import torch

    from fl.adapters import make_adapter
    from fl.archspec import SPECS, build_torch

    spec = SPECS[model_name]
    net = build_torch(spec)
    make_adapter("torch", spec).from_canonical(net, weights)
    net.eval()
    with torch.no_grad():
        nchw = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
        return net(nchw).numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--framework", choices=("tf", "torch", "both"), default="tf")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    weights, header = load_checkpoint(args.checkpoint)
    model_name = header["model"]
    LOGGER.info(
        "checkpoint: model=%s tensors=%d dataset=%s",
        model_name,
        header["num_tensors"],
        _DATASET_FOR_MODEL.get(model_name, "?"),
    )

    x, y = _test_samples(model_name, args.samples, args.seed)

    outputs = {}
    if args.framework in ("torch", "both"):
        outputs["torch"] = predict_torch(weights, model_name, x)
    if args.framework in ("tf", "both"):
        outputs["tf"] = predict_tf(weights, model_name, x)

    for name, probs in outputs.items():
        picks = np.argmax(probs, axis=1)
        accuracy = float(np.mean(picks == y))
        LOGGER.info(
            "%s: predicted %s | true %s | sample accuracy %.2f",
            name,
            picks.tolist(),
            y.tolist(),
            accuracy,
        )

    if len(outputs) == 2:
        agree = np.argmax(outputs["tf"], axis=1) == np.argmax(outputs["torch"], axis=1)
        LOGGER.info("cross-framework argmax agreement: %d/%d", int(agree.sum()), len(agree))
        if not agree.all():
            LOGGER.error("frameworks disagree on %d samples", int((~agree).sum()))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
