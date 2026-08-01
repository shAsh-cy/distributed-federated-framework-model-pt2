"""Capability discovery: what this deployment can run.

Derived from the fl registries at call time, so the frontend hardcodes
nothing — a dataset or architecture added to fl/ appears here without a
coordinator change.
"""

from __future__ import annotations


def datasets() -> list[dict]:
    from fl.config import DATASET_MODEL, VALID_DATASETS
    from fl.data import DATASET_NUM_CLASSES

    partition_schemes = {
        "fashion_mnist": ["iid", "dirichlet"],
        "femnist": ["natural"],
    }
    return [
        {
            "name": name,
            "num_classes": DATASET_NUM_CLASSES[name],
            "model": DATASET_MODEL[name],
            "partition_schemes": partition_schemes.get(name, []),
        }
        for name in VALID_DATASETS
    ]


def algorithms() -> list[dict]:
    return [
        {
            "name": "fedavg",
            "description": "Sample-count weighted federated averaging.",
            "differentially_private": False,
        },
        {
            "name": "dp-fedavg",
            "description": (
                "Client-level DP FedAvg: per-client update clipping and Gaussian "
                "noise at the aggregator via TFF; epsilon computed by the RDP "
                "accountant, never chosen."
            ),
            "differentially_private": True,
        },
    ]


def architectures() -> list[dict]:
    from fl.archspec import SPECS

    return [
        {
            "name": name,
            "input_shape": list(spec.input_shape),
            "parameter_count": spec.parameter_count(),
            "tensors": [
                {"name": n, "shape": list(s)}
                for n, s in zip(spec.canonical_names(), spec.canonical_shapes(), strict=True)
            ],
        }
        for name, spec in SPECS.items()
    ]
