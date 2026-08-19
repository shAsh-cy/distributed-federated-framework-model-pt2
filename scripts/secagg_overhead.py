"""What secure aggregation costs, measured — bytes, masking compute, wall-clock.

Four phases, each written to its own JSON under docs/ and skipped if already
present (so a crashed run resumes in place):

* ``bytes``     the analytic communication_cost model at m in {5, 10, 20} (and a
                curve to 200), split into the 2x update inflation and the O(m^2)
                key/share setup. Pure; no TF, no gRPC.
* ``masking``   the client-side masking compute isolated from training: time to
                mask one 225k-word update against m-1 peers, across m. This is
                where the O(m^2) total (O(m) per client) shows up in CPU time
                rather than only in bytes. Pure; numpy only.
* ``walltime``  end-to-end seconds per round, secure vs plain FedAvg, over real
                in-process gRPC with real training, at m in {5, 10, 20}. This is
                the ~1h part.
* ``dropout``   the added wall-clock of one induced mid-round dropout and its
                Shamir recovery, against a clean secure round at the same m.

The bytes and masking phases run anywhere; walltime and dropout need the TF/gRPC
stack (the Docker image). See docs/secure_aggregation.md for the write-up this
feeds. Not launched by hand — scripts/run_secagg_overhead_batch.sh queues it
behind the batch lock; see that script.

    python scripts/secagg_overhead.py --phases bytes masking walltime dropout \
        --out-dir docs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fl.secure_aggregation import communication_cost  # noqa: E402

LOGGER = logging.getLogger("secagg_overhead")

SMALL_CNN_PARAMS = 225_034
COHORTS = (5, 10, 20)
CURVE = (5, 10, 20, 50, 100, 200)


# ---------------------------------------------------------------------------
# Phase: bytes (pure)
# ---------------------------------------------------------------------------


def measure_bytes() -> dict:
    """The analytic byte model, separated into its two growth regimes."""
    rows = []
    for m in CURVE:
        c = communication_cost(num_clients=m, num_params=SMALL_CNN_PARAMS)
        b = c["breakdown"]
        setup = b["registration"] + b["roster_broadcast"] + b["share_distribution"]
        rows.append(
            {
                "num_clients": m,
                "secure_total_bytes": c["secure_total_bytes"],
                "plain_total_bytes": c["plain_total_bytes"],
                "overhead_ratio": c["overhead_ratio"],
                "mask_setup_bytes": setup,  # registration + roster + shares (O(m^2))
                "share_distribution_bytes": b["share_distribution"],
                "masked_update_bytes": b["masked_updates"],
            }
        )
    return {
        "num_params": SMALL_CNN_PARAMS,
        "note": (
            "The 2x floor is uint64 words vs float32; the drift above 2.0 is the "
            "O(m^2) key-exchange and share-distribution setup. SecAgg+ (sparse "
            "neighbour graphs, what Flower ships) is the production answer to the "
            "quadratic term."
        ),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Phase: masking (pure)
# ---------------------------------------------------------------------------


def measure_masking(repeats: int = 3) -> dict:
    """Time the client-side masking of one full-size update against m-1 peers.

    Uses the real ParticipantSession, so this is the exact per-client cost the
    live path pays: one self mask plus m-1 pairwise masks over a 225k-word
    vector. Per client it is O(m); summed over the cohort it is the O(m^2) the
    pairwise scheme is known for. No training, no network — just the masking."""
    from fl.secure_round import ParticipantSession, RosterEntry

    rng = np.random.default_rng(0)
    values = rng.normal(size=SMALL_CNN_PARAMS).astype(np.float32)

    rows = []
    for m in CURVE:
        parts = [ParticipantSession(f"c{i}", i) for i in range(m)]
        roster = {p.client_id: RosterEntry(p.order, p.public_key) for p in parts}
        for p in parts:
            p.make_share_parcels(roster, threshold=m // 2 + 1)
        # Time one client masking against the other m-1 peers.
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            parts[0].masked_update(values, weight=5000.0)
            best = min(best, time.perf_counter() - t0)
        rows.append(
            {
                "num_clients": m,
                "per_client_mask_seconds": best,
                "cohort_mask_seconds_estimate": best * m,  # O(m^2) total
            }
        )
        LOGGER.info("masking m=%d: %.4fs/client", m, best)
    return {"num_params": SMALL_CNN_PARAMS, "repeats": repeats, "rows": rows}


# ---------------------------------------------------------------------------
# Phase: walltime (TF + gRPC)
# ---------------------------------------------------------------------------


def _run_inprocess(secure: bool, num_clients: int, rounds: int, seed: int = 42) -> list[float]:
    """Run one experiment in-process over real gRPC and return per-round seconds.

    Plain uses FederatedServer + FederatedClient; secure uses their V3 twins.
    Identical model, data and hyperparameters, so the per-round difference is the
    secure-aggregation overhead and nothing else."""
    import random
    import threading

    import tensorflow as tf

    from fl.client import FederatedClient
    from fl.config import Config
    from fl.data import load_federated
    from fl.models import build_model
    from fl.secure_client import SecureFederatedClient
    from fl.secure_server import SecureFederatedServer
    from fl.server import FederatedServer, build_evaluator

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg = Config.from_dict(
        {
            "seed": seed,
            "data": {"num_clients": num_clients, "partition": "iid"},
            "model": {"name": "small_cnn"},
            "training": {
                "rounds": rounds,
                "client_fraction": 1.0,
                "local_epochs": 1,
                "batch_size": 32,
            },
            "server": {
                "host": "127.0.0.1",
                "port": _free_port(),
                "round_deadline_seconds": 120.0,
                "min_clients_per_round": 2,
                "registration_timeout_seconds": 120.0,
            },
        }
    )
    train, test, shards = load_federated(cfg.data, seed=seed)
    initial = build_model(cfg.model.name, seed=seed).get_weights()
    evaluate_fn = build_evaluator(cfg.model.name, test.x, test.y)

    if secure:
        server = SecureFederatedServer(cfg, initial, evaluate_fn)
        client_cls = SecureFederatedClient
    else:
        from fl.aggregation import FedAvgAggregator

        server = FederatedServer(cfg, initial, FedAvgAggregator(), evaluate_fn)
        client_cls = FederatedClient
    server.start()
    address = f"127.0.0.1:{cfg.server.port}"

    clients = []
    for i in range(num_clients):
        c = client_cls(cfg, address, desired_client_id=f"client-{i}")
        c.register()
        c.load_data(train=train, shards=shards)
        clients.append(c)
    threads = [
        threading.Thread(target=c.run, kwargs={"poll_interval": 0.05}, daemon=True) for c in clients
    ]
    try:
        for t in threads:
            t.start()
        metrics = server.run_rounds()
        for t in threads:
            t.join(timeout=60)
    finally:
        for c in clients:
            c.close()
        server.stop(grace=0)
    return [round(m.duration_seconds, 4) for m in metrics]


def measure_walltime(rounds: int = 3) -> dict:
    rows = []
    for m in COHORTS:
        plain = _run_inprocess(secure=False, num_clients=m, rounds=rounds)
        secure = _run_inprocess(secure=True, num_clients=m, rounds=rounds)
        rows.append(
            {
                "num_clients": m,
                "rounds": rounds,
                "plain_seconds_per_round": plain,
                "secure_seconds_per_round": secure,
                "plain_mean": float(np.mean(plain)) if plain else None,
                "secure_mean": float(np.mean(secure)) if secure else None,
                "overhead_ratio": (
                    float(np.mean(secure) / np.mean(plain)) if plain and secure else None
                ),
            }
        )
        LOGGER.info(
            "walltime m=%d: plain=%.2fs secure=%.2fs",
            m,
            float(np.mean(plain)),
            float(np.mean(secure)),
        )
    return {"rounds": rounds, "rows": rows}


# ---------------------------------------------------------------------------
# Phase: dropout (TF + gRPC)
# ---------------------------------------------------------------------------


def measure_dropout(num_clients: int = 10, rounds: int = 2) -> dict:
    """One clean secure run vs one where a single client drops after sharing, at
    the same m. The difference is what a mid-round dropout and its recovery cost.
    Reuses the integration-style drop client from the test path."""
    import random
    import threading

    import tensorflow as tf

    from fl.config import Config
    from fl.data import load_federated
    from fl.models import build_model
    from fl.secure_client import SecureFederatedClient
    from fl.secure_round import ParticipantSession
    from fl.secure_server import SecureFederatedServer
    from fl.server import build_evaluator

    class _DropOnce(SecureFederatedClient):
        def _participate(self, response):
            if response.round == 1:
                roster, my_order = self._roster_from(response)
                p = ParticipantSession(self.client_id, my_order, seed=self._mask_seed)
                self._send_shares(p, roster, int(response.threshold), response.round)
                return
            super()._participate(response)

    def one_run(drop: bool) -> list[float]:
        seed = 7
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        cfg = Config.from_dict(
            {
                "seed": seed,
                "data": {"num_clients": num_clients, "partition": "iid"},
                "model": {"name": "small_cnn"},
                "training": {"rounds": rounds, "client_fraction": 1.0, "batch_size": 32},
                "server": {
                    "host": "127.0.0.1",
                    "port": _free_port(),
                    "round_deadline_seconds": 30.0,
                    "min_clients_per_round": 2,
                    "registration_timeout_seconds": 120.0,
                },
            }
        )
        train, test, shards = load_federated(cfg.data, seed=seed)
        initial = build_model(cfg.model.name, seed=seed).get_weights()
        server = SecureFederatedServer(
            cfg, initial, build_evaluator(cfg.model.name, test.x, test.y)
        )
        server.start()
        address = f"127.0.0.1:{cfg.server.port}"
        clients = []
        for i in range(num_clients):
            cls = _DropOnce if (drop and i == num_clients - 1) else SecureFederatedClient
            c = cls(cfg, address, desired_client_id=f"client-{i}")
            c.register()
            c.load_data(train=train, shards=shards)
            clients.append(c)
        threads = [
            threading.Thread(target=c.run, kwargs={"poll_interval": 0.05}, daemon=True)
            for c in clients
        ]
        try:
            for t in threads:
                t.start()
            metrics = server.run_rounds()
            for t in threads:
                t.join(timeout=60)
        finally:
            for c in clients:
                c.close()
            server.stop(grace=0)
        return [round(m.duration_seconds, 4) for m in metrics]

    clean = one_run(drop=False)
    dropped = one_run(drop=True)
    return {
        "num_clients": num_clients,
        "rounds": rounds,
        "clean_seconds_per_round": clean,
        "with_dropout_seconds_per_round": dropped,
        "note": (
            "Round 1 of the dropout run pays a deadline wait for the missing masked "
            "update plus a Shamir recovery; round 2 is a clean secure round again."
        ),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


_PHASES = {
    "bytes": measure_bytes,
    "masking": measure_masking,
    "walltime": measure_walltime,
    "dropout": measure_dropout,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", nargs="+", choices=sorted(_PHASES), default=sorted(_PHASES))
    parser.add_argument("--out-dir", default="docs")
    parser.add_argument("--force", action="store_true", help="recompute even if the JSON exists")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for phase in args.phases:
        out = out_dir / f"_secagg_overhead_{phase}.json"
        if out.exists() and not args.force:
            LOGGER.info("phase %s already done (%s); skipping", phase, out)
            continue
        LOGGER.info("running phase %s", phase)
        result = _PHASES[phase]()
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
