"""The message-driven orchestration equals the in-process protocol, exactly.

If run_round (the distributed sequencing, with an in-memory transport) produces
the same aggregate and the same byte accounting as run_secure_round (the
in-process reference), then the gRPC layer — which makes these same message
exchanges — is transporting a correct protocol, and any live-path bug is in the
wire, not the orchestration. numpy + stdlib only.
"""

from __future__ import annotations

import numpy as np
import pytest

from fl.secure_aggregation import (
    InsufficientSharesError,
    SecureAggregationError,
    communication_cost,
    run_secure_round,
)
from fl.secure_round import ParticipantSession, RosterEntry, ServerSession, run_round


def _updates(n: int, size: int = 40, seed: int = 11) -> list[tuple[str, np.ndarray, float]]:
    rng = np.random.default_rng(seed)
    return [
        (f"c{i}", rng.normal(scale=0.5, size=size).astype(np.float32), float(1000 + 500 * i))
        for i in range(n)
    ]


class TestEquivalenceToInProcess:
    def test_clean_round_matches_run_secure_round(self):
        updates = _updates(5)
        messaged, m_report = run_round(updates, threshold=3)
        reference, r_report = run_secure_round(updates, threshold=3)
        np.testing.assert_array_equal(messaged, reference)
        assert m_report["weight_sum"] == pytest.approx(r_report["weight_sum"])
        assert m_report["dropped"] == r_report["dropped"]

    def test_byte_accounting_matches_the_analytic_model(self):
        """The distributed routing logs shares and reveals the same way the
        in-process server does, so communication_cost still reconciles to the
        byte — the property tests/test_secure_aggregation.py asserts, now over
        the message path."""
        updates = _updates(6, size=32)
        _, report = run_round(
            updates, threshold=3, drop_before_submit={"c0"}, drop_during_recovery={"c1"}
        )
        model = communication_cost(num_clients=6, num_params=32, dropouts=1, recovery_silent=1)
        assert report["total_bytes"] == model["secure_total_bytes"]

    def test_dropout_before_submit_matches(self):
        updates = _updates(5)
        messaged, report = run_round(updates, threshold=3, drop_before_submit={"c2"})
        reference, _ = run_secure_round(updates, threshold=3, drop_before_submit={"c2"})
        np.testing.assert_array_equal(messaged, reference)
        assert report["dropped"] == ["c2"]

    def test_dropout_during_recovery_matches(self):
        updates = _updates(6)
        messaged, report = run_round(
            updates, threshold=3, drop_before_submit={"c0"}, drop_during_recovery={"c1"}
        )
        reference, _ = run_secure_round(
            updates, threshold=3, drop_before_submit={"c0"}, drop_during_recovery={"c1"}
        )
        np.testing.assert_array_equal(messaged, reference)
        assert "c1" in report["survivors"]
        assert "c1" not in report["responders"]

    def test_below_threshold_aborts_explicitly(self):
        updates = _updates(4)
        with pytest.raises(InsufficientSharesError, match="threshold"):
            run_round(updates, threshold=4, drop_during_recovery={"c3"})


class TestSessionGuards:
    def test_either_or_reveal_rule_is_enforced_across_the_message_api(self):
        """A client asked (by a malicious server) to reveal both secrets of one
        owner refuses — the guarantee survives the distributed rephrasing."""
        p = ParticipantSession("c0", 0)
        roster = {
            "c0": RosterEntry(0, p.public_key),
            "c1": RosterEntry(1, ParticipantSession("c1", 1).public_key),
        }
        p.make_share_parcels(roster, threshold=2)
        # It holds its own shares after routing in a real round; here it at least
        # refuses the both-secrets request for any owner it does hold.
        p2 = ParticipantSession("c1", 1)
        roster2 = {"c0": roster["c0"], "c1": RosterEntry(1, p2.public_key)}
        parcels = p2.make_share_parcels(roster2, threshold=2)
        p.receive_shares(parcels["c0"])
        p.reveal([("c1", "self_mask")])
        with pytest.raises(SecureAggregationError, match="both secrets"):
            p.reveal([("c1", "key_seed")])

    def test_accept_reveals_rejects_an_unrequested_responder(self):
        updates = _updates(4)
        participants = {
            cid: ParticipantSession(cid, order) for order, (cid, _, _) in enumerate(updates)
        }
        server = ServerSession(threshold=3)
        for p in participants.values():
            server.announce(p.client_id, p.order, p.public_key)
        roster = server.roster_broadcast()
        for p in participants.values():
            server.route_shares(p.make_share_parcels(roster, 3))
        for p in participants.values():
            p.receive_shares(server.shares_for(p.client_id))
        for cid, values, weight in updates:
            server.submit_masked(cid, participants[cid].masked_update(values, weight))
        server.reveal_requests({"c0", "c1", "c2"})  # c3 not a responder
        with pytest.raises(SecureAggregationError, match="not asked to reveal"):
            server.accept_reveals("c3", participants["c3"].reveal([("c0", "self_mask")]))
