"""Run one federated experiment end to end and write its metrics to JSON.

Seeded throughout: Python, NumPy and TensorFlow RNGs are set from ``config.seed``,
the data partition is derived from it, the initial global model is built from it,
and the server's client sampling uses it. Two runs of the same config on the same
machine produce the same partition, the same initial weights and the same cohort
sequence.

Clients run as threads in this process against a real gRPC server on localhost --
the same code path the Docker deployment uses, minus container overhead. Data is
loaded once and shared, because ten clients each loading Fashion-MNIST would cost
~1.9 GB for no benefit.

Note on exact reproducibility: with DP enabled the Gaussian noise is drawn inside
TFF's aggregation process, and thread interleaving determines which clients report
first, so DP runs are reproducible in configuration but not bit-identical in
outcome. Non-DP accuracy is deterministic up to floating-point summation order.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml \
        --metrics-out results/default.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fl.client import FederatedClient  # noqa: E402
from fl.config import Config  # noqa: E402
from fl.data import load_fashion_mnist, partition, partition_summary  # noqa: E402
from fl.models import build_model, count_parameters  # noqa: E402
from fl.server import FederatedServer, build_evaluator  # noqa: E402

LOGGER = logging.getLogger("experiment")


def seed_everything(seed: int) -> None:
    """Seed every RNG that can affect the run."""
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run(config: Config, port: int | None = None) -> dict:
    """Run every round of ``config`` and return a JSON-serialisable report."""
    from fl.aggregation import compute_epsilon, make_aggregator

    seed_everything(config.seed)

    train, test = load_fashion_mnist()
    shards = partition(
        train.y,
        num_clients=config.data.num_clients,
        scheme=config.data.partition,
        alpha=config.data.dirichlet_alpha,
        seed=config.seed,
    )
    summary = partition_summary(train.y, shards)
    LOGGER.info(
        "partition (%s): shard sizes %s",
        config.data.partition,
        [row["num_examples"] for row in summary],
    )

    template = build_model(config.model.name, seed=config.seed)
    initial_weights = template.get_weights()
    num_params = count_parameters(template)

    aggregator = make_aggregator(
        dp_enabled=config.privacy.enabled,
        noise_multiplier=config.privacy.noise_multiplier,
        l2_clip_norm=config.privacy.l2_clip_norm,
        clients_per_round=config.clients_per_round,
    )

    epsilon_fn = None
    if config.privacy.enabled:

        def epsilon_fn(completed_rounds: int) -> float:
            return compute_epsilon(
                noise_multiplier=config.privacy.noise_multiplier,
                sampling_rate=config.client_sampling_rate,
                rounds=completed_rounds,
                delta=config.privacy.delta,
            )

    # Bind an ephemeral port unless one was requested, so concurrent experiments
    # do not collide.
    if port is None:
        port = free_port()
    config = config.replace(server={"host": "127.0.0.1", "port": port})

    server = FederatedServer(
        config=config,
        initial_weights=initial_weights,
        aggregator=aggregator,
        evaluate_fn=build_evaluator(config.model.name, test.x, test.y),
        epsilon_fn=epsilon_fn,
    )
    server.start()
    address = f"127.0.0.1:{config.server.port}"

    baseline_loss, baseline_accuracy = server.evaluate_fn(initial_weights)
    LOGGER.info("round 0 (untrained): acc=%.4f loss=%.4f", baseline_accuracy, baseline_loss)

    clients: list[FederatedClient] = []
    threads: list[threading.Thread] = []
    started = time.monotonic()
    try:
        for i in range(config.data.num_clients):
            client = FederatedClient(config, address, desired_client_id=f"client-{i}")
            client.register()
            client.load_data(train=train, shards=shards)
            clients.append(client)

        for client in clients:
            t = threading.Thread(target=client.run, kwargs={"poll_interval": 0.1}, daemon=True)
            t.start()
            threads.append(t)

        metrics = server.run_rounds()
        for t in threads:
            t.join(timeout=60)
    finally:
        for client in clients:
            client.close()
        server.stop()

    wall_clock = time.monotonic() - started
    final = metrics[-1] if metrics else None

    return {
        "config": config.to_dict(),
        "aggregator": aggregator.name,
        "model": {"name": config.model.name, "parameters": num_params},
        "partition": summary,
        "privacy": {
            "enabled": config.privacy.enabled,
            "noise_multiplier": config.privacy.noise_multiplier,
            "l2_clip_norm": config.privacy.l2_clip_norm,
            "delta": config.privacy.delta,
            "client_sampling_rate": config.client_sampling_rate,
            "clients_per_round": config.clients_per_round,
            "epsilon": epsilon_fn(config.training.rounds) if epsilon_fn else None,
        },
        "baseline_untrained": {"accuracy": baseline_accuracy, "loss": baseline_loss},
        "final": {
            "accuracy": final.accuracy if final else None,
            "loss": final.loss if final else None,
            "epsilon": final.epsilon if final else None,
        },
        "wall_clock_seconds": round(wall_clock, 2),
        "total_bytes_sent": sum(m.bytes_sent for m in metrics),
        "total_bytes_received": sum(m.bytes_received for m in metrics),
        "total_dropped_client_rounds": sum(m.num_dropped for m in metrics),
        "rounds": [m.to_dict() for m in metrics],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = Config.from_yaml(args.config)
    report = run(config)

    final = report["final"]
    LOGGER.info(
        "FINAL accuracy=%.4f loss=%.4f epsilon=%s (%.1fs)",
        final["accuracy"],
        final["loss"],
        f"{final['epsilon']:.3f}" if final["epsilon"] is not None else "n/a (no DP)",
        report["wall_clock_seconds"],
    )

    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
