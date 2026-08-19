"""The secure-aggregation client: trains as usual, then masks before sending.

:class:`SecureFederatedClient` reuses :class:`fl.client.FederatedClient` for data
loading and local training and changes only what leaves the machine: instead of
posting plaintext weights, it derives pairwise masks per the tested protocol
(:mod:`fl.secure_round`) and posts masked uint64 words. The masking arithmetic
lives in :mod:`fl.secure_aggregation`; this class is the client half of the V3
transport.

Its Diffie-Hellman keypair is derived once, at construction, from a random seed
and announced at registration — so the server holds the client's public key
before any cohort is sampled, and a round's roster is a subset it already has.
The same seed rebuilds the participant each round, so the key the roster lists
always matches the key that masks.
"""

from __future__ import annotations

import argparse
import logging
import secrets
import time

import grpc
import numpy as np

from .client import FederatedClient
from .config import Config
from .proto import fl_comm_pb2
from .secure_aggregation import SecureClient, Share
from .secure_live import flatten_weights
from .secure_round import ParticipantSession, RosterEntry

LOGGER = logging.getLogger("fl.secure_client")

_PB = fl_comm_pb2

# The public key is a 2048-bit group element; the wire carries a fixed 256 bytes.
_PUBLIC_KEY_BYTES = 256
_SHARE_VALUE_BYTES = 66


class SecureFederatedClient(FederatedClient):
    """A federated client that submits masked updates over the V3 secure path."""

    def __init__(
        self,
        config: Config,
        server_address: str,
        desired_client_id: str = "",
        framework: str = "tensorflow",
        mask_seed: bytes | None = None,
    ) -> None:
        super().__init__(config, server_address, desired_client_id, framework)
        # One keypair for the client's whole lifetime: the seed rebuilds it each
        # round, so the announced public key and the masking key are always the
        # same. secrets.token_bytes gives a fresh, unguessable seed per client.
        self._mask_seed = mask_seed if mask_seed is not None else secrets.token_bytes(32)
        self._public_key = SecureClient("_probe", 0, seed=self._mask_seed).public_key
        self._attempted: set[tuple[int, int]] = set()

    def register(self) -> str:
        """Register as a V3 client, announcing the masking public key."""
        response = self.stub.Register(
            _PB.RegisterRequest(
                protocol_version=_PB.PROTOCOL_VERSION_V3,
                desired_client_id=self.desired_client_id,
                framework=self.framework,
                masking_public_key=self._public_key.to_bytes(_PUBLIC_KEY_BYTES, "big"),
            )
        )
        from .client import RegistrationError

        if not response.accepted:
            raise RegistrationError(f"server refused registration: {response.rejection_reason}")
        self.client_id = response.client_id
        self.shard_index = response.shard_index
        LOGGER.info(
            "registered as %s (secure), shard %d of %d",
            self.client_id,
            self.shard_index,
            response.num_clients,
        )
        return self.client_id

    # -- one secure round ---------------------------------------------------

    def _participate(self, response) -> None:
        """Train, mask, submit, and take part in recovery for one round."""
        roster, my_order = self._roster_from(response)
        threshold = int(response.threshold)
        participant = ParticipantSession(self.client_id, my_order, seed=self._mask_seed)

        # Local training on the published global weights (same trainer as plain).
        weights, num_examples, loss, accuracy = self.train_one_round(response)

        # Share both secrets among the cohort; the server routes them. This
        # happens BEFORE the masked submit precisely so that a client which then
        # drops has already distributed the key-seed shares survivors need to
        # cancel its orphaned pairwise masks.
        self._send_shares(participant, roster, threshold, response.round)

        # Mask the trained weights (scaled by the example count) and submit.
        flat, _ = flatten_weights(weights)
        words = participant.masked_update(flat, float(num_examples))
        masked = np.ascontiguousarray(words.astype("<u8"))
        reply = self.stub.SubmitMaskedUpdate(
            _PB.SubmitMaskedUpdateRequest(
                client_id=self.client_id,
                round=response.round,
                model_version=response.model_version,
                masked_words=masked.tobytes(),
                num_words=int(words.size),
            )
        )
        if reply.status != _PB.UPDATE_STATUS_ACCEPTED:
            LOGGER.warning(
                "client %s: masked update rejected (%s); skipping recovery",
                self.client_id,
                _PB.UpdateStatus.Name(reply.status),
            )
            return
        LOGGER.info(
            "client %s: round %d masked update accepted (n=%d, loss=%.4f, acc=%.4f)",
            self.client_id,
            response.round,
            num_examples,
            loss,
            accuracy,
        )
        self._recover(participant, response.round)

    def _send_shares(
        self,
        participant: ParticipantSession,
        roster: dict[str, RosterEntry],
        threshold: int,
        round_index: int,
    ) -> None:
        """Shamir-share both secrets among the cohort and upload the parcels for
        the server to route to their holders."""
        parcels = participant.make_share_parcels(roster, threshold)
        share_req = _PB.SubmitSecureSharesRequest(client_id=self.client_id, round=round_index)
        for recipient_id, shares in parcels.items():
            parcel = share_req.parcels.add(recipient_id=recipient_id)
            for share in shares:
                parcel.shares.add(
                    owner=share.owner,
                    kind=share.kind,
                    index=int(share.index),
                    value=int(share.value).to_bytes(_SHARE_VALUE_BYTES, "big"),
                )
        self.stub.SubmitSecureShares(share_req)

    def _recover(self, participant: ParticipantSession, round_index: int) -> None:
        """Poll for the reveal request, then reveal the shares recovery needs."""
        deadline = time.monotonic() + self.config.server.round_deadline_seconds
        while time.monotonic() < deadline:
            reveal = self.stub.GetRevealShares(
                _PB.GetRevealRequest(client_id=self.client_id, round=round_index)
            )
            if reveal.ready:
                break
            time.sleep(0.1)
        else:
            LOGGER.warning(
                "client %s: recovery for round %d never opened before its deadline",
                self.client_id,
                round_index,
            )
            return

        participant.receive_shares(
            [
                Share(
                    owner=s.owner,
                    kind=s.kind,
                    index=int(s.index),
                    value=int.from_bytes(s.value, "big"),
                )
                for s in reveal.held_shares
            ]
        )
        needed = [(item.owner, item.kind) for item in reveal.needed]
        shares = participant.reveal(needed)
        req = _PB.SubmitRevealsRequest(client_id=self.client_id, round=round_index)
        for share in shares:
            req.shares.add(
                owner=share.owner,
                kind=share.kind,
                index=int(share.index),
                value=int(share.value).to_bytes(_SHARE_VALUE_BYTES, "big"),
            )
        self.stub.SubmitReveals(req)
        LOGGER.info(
            "client %s: revealed %d shares for round %d", self.client_id, len(shares), round_index
        )

    def _roster_from(self, response) -> tuple[dict[str, RosterEntry], int]:
        roster: dict[str, RosterEntry] = {}
        my_order: int | None = None
        for entry in response.roster:
            order = int(entry.order)
            roster[entry.client_id] = RosterEntry(
                order=order, public_key=int.from_bytes(entry.public_key, "big")
            )
            if entry.client_id == self.client_id:
                my_order = order
        if my_order is None:
            raise RuntimeError(f"client {self.client_id!r} absent from its own round roster")
        return roster, my_order

    # -- main loop ----------------------------------------------------------

    def run(
        self,
        poll_interval: float = 0.5,
        max_idle_polls: int = 100_000,
        max_unreachable_polls: int = 20,
    ) -> None:
        """Poll the secure round until the server stops, or goes away."""
        if self.client_id is None:
            self.register()
        if self.x is None:
            self.load_data()

        idle = 0
        unreachable = 0
        while True:
            try:
                response = self.stub.GetSecureRound(
                    _PB.GetSecureRoundRequest(client_id=self.client_id)
                )
                unreachable = 0
            except ValueError:
                LOGGER.info("client %s: channel closed, exiting", self.client_id)
                return
            except grpc.RpcError as exc:
                unreachable += 1
                if unreachable > max_unreachable_polls:
                    LOGGER.info(
                        "client %s: server unreachable after %d attempts (%s); exiting",
                        self.client_id,
                        unreachable,
                        exc.code(),
                    )
                    return
                time.sleep(poll_interval)
                continue

            if response.action == _PB.ROUND_ACTION_STOP:
                LOGGER.info("client %s: server signalled stop", self.client_id)
                return

            attempt_key = (response.round, response.model_version)
            if response.action != _PB.ROUND_ACTION_TRAIN or attempt_key in self._attempted:
                idle += 1
                if idle > max_idle_polls:
                    LOGGER.warning("client %s: idle limit reached, exiting", self.client_id)
                    return
                time.sleep(poll_interval)
                continue

            idle = 0
            self._attempted.add(attempt_key)
            try:
                self._participate(response)
            except grpc.RpcError as exc:
                # A mid-round RPC failure degrades this client to a dropout for
                # the round: if it had already shared, survivors recover its
                # masks; if not, the threshold check aborts the round cleanly.
                # Either way the client stays alive for the next round rather
                # than killing its thread on a transient error.
                LOGGER.warning(
                    "client %s: round %d participation failed (%s); treating as a dropout",
                    self.client_id,
                    response.round,
                    exc.code(),
                )
            time.sleep(poll_interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Secure-aggregation federated client.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--server", default=None)
    parser.add_argument("--client-id", default="")
    parser.add_argument("--framework", default="tensorflow", choices=("tensorflow", "torch"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.framework == "torch":
        import torch  # noqa: F401

    config = Config.from_yaml(args.config)
    address = args.server or f"{config.server.host}:{config.server.port}"
    client = SecureFederatedClient(
        config, address, desired_client_id=args.client_id, framework=args.framework
    )
    try:
        client.register()
        client.load_data()
        client.run()
    finally:
        client.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
