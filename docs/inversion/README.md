# Gradient-inversion figures

These grids are **generated**, not committed by hand, by
[scripts/inversion_demo.py](../../scripts/inversion_demo.py). They are produced in
the overnight batch (queued behind the shared lock by
[scripts/run_inversion_batch.sh](../../scripts/run_inversion_batch.sh)), not on a
laptop, because the reconstructions run thousands of attack iterations over the
225k-parameter model. Everything is deterministic under `--seed`.

What lands here when the batch runs:

| File | What it shows |
|---|---|
| `batch_size_1.png`, `batch_size_4.png`, `batch_size_8.png` | Reconstruction with no defence at batch sizes 1 / 4 / 8 — clean at 1, degrading as gradients average |
| `defense_curve.png` | The same batch-of-one attack against a gradient with client-side DP noise at multipliers {0, 0.3, 2.0}, each panel labelled with the ε that multiplier buys at the working config — recognisable garment → smeared → mush |
| `fedavg_e10_failure.png` | One attempt against a realistic E=10 FedAvg update — it **fails**, and that failure is the honest headline result |
| `summary.json` | Reconstruction MSE and final objective for every panel |

Read these as statements about the **threat model** that motivates DP and secure
aggregation, not as measured production leakage — see the framing in
[docs/privacy_threats.md](../privacy_threats.md). The `fedavg_e10_failure.png`
panel is the load-bearing one: under realistic multi-epoch FedAvg the attack
does not reconstruct.

A 30-second, hermetic version of the attack runs in CI as
[tests/test_inversion_smoke.py](../../tests/test_inversion_smoke.py) — it verifies
the pipeline is deterministic and makes progress, without the full-length grids.
