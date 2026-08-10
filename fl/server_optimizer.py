"""Server optimizers: the FedOpt family (Reddi et al., ICLR 2021).

FedAvg applies the round's aggregated client delta to the global model
directly: ``w <- w + delta``. The FedOpt reframing treats that delta as a
*pseudo-gradient* -- one noisy first-order signal about where the population
loss decreases -- and hands it to a server-side optimizer that decides the
actual step:

    FedAvg    identity          w <- w + 1.0 * delta          (no state)
    FedAvgM   momentum          v <- beta_m*v + delta;         w <- w + lr*v
    FedAdam   Adam moments      m, v as Adam over delta;       w <- w + lr*m/(sqrt(v)+tau)
    FedYogi   Yogi moments      Adam's m, Yogi's v;            same step rule

Two deliberate departures from textbook Adam, both from Algorithm 2 of the
paper (arXiv:2003.00295) and both load-bearing for reproducing its results:

* **No bias correction.** The paper's server optimizers omit the
  ``1/(1-beta^t)`` warm-up terms; ``tau`` and the ``v`` initialisation play
  that role instead.
* **``v`` initialises to ``tau**2``,** not zero, satisfying the paper's
  ``v_{-1} >= tau^2 > 0`` requirement -- so the very first step is already
  bounded by ``lr * |m| / (2*tau)`` rather than exploding on a tiny ``v``.

Yogi differs from Adam only in the second moment. Adam interpolates
geometrically toward ``delta^2``:

    v <- beta2*v + (1-beta2)*delta^2

so a burst of large deltas inflates ``v`` multiplicatively fast, and the
effective learning rate collapses just as quickly. Yogi moves *additively*,
always by exactly ``(1-beta2)*delta^2``, in whichever direction closes the
gap to ``delta^2``:

    v <- v - (1-beta2) * delta^2 * sign(v - delta^2)

The two coincide whenever ``v`` is already at ``delta^2`` and differ
everywhere else; the test suite pins the difference on both sides of the
sign. (Zaheer et al. 2018 introduced Yogi for exactly the federated-style
setting where sparse, bursty pseudo-gradients make Adam's ``v`` collapse.)

State lives *here*, not per round: momentum and moments must persist across
rounds to mean anything, which is why the server optimizer is owned by the
aggregator (constructed once per run) rather than being rebuilt each round.
All state accumulates in float64; the aggregator casts the applied result
back to float32 at the model boundary.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

Weights = list[np.ndarray]

VALID_SERVER_OPTIMIZERS: tuple[str, ...] = ("fedavg", "fedavgm", "fedadam", "fedyogi")


class ServerOptimizer(Protocol):
    """One step per round: pseudo-gradient in, weight update out."""

    #: Recorded in metrics via the owning aggregator's name.
    name: str

    def step(self, delta: Weights) -> Weights:
        """Map the round's aggregated delta to the update actually applied."""
        ...

    def reset(self) -> None:
        """Drop all cross-round state, as a fresh construction would."""
        ...


class SGDServerOptimizer:
    """Server SGD with momentum: FedAvgM, and plain FedAvg as its identity case.

    ``v <- momentum * v + delta;  update = learning_rate * v``

    With ``momentum=0`` and ``learning_rate=1.0`` this is exactly FedAvg --
    no state survives a round and the delta passes through untouched. The
    equivalence is asserted bit-exactly in the tests, which is what licenses
    implementing FedAvg and FedAvgM as one class.
    """

    def __init__(self, learning_rate: float = 1.0, momentum: float = 0.0) -> None:
        if learning_rate <= 0:
            raise ValueError(f"server learning_rate must be > 0, got {learning_rate}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"server momentum must be in [0, 1), got {momentum}")
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.name = "fedavgm" if momentum > 0.0 else "fedavg"
        self._velocity: list[np.ndarray] | None = None

    def step(self, delta: Weights) -> Weights:
        d = [np.asarray(t, dtype=np.float64) for t in delta]
        if self._velocity is None:
            self._velocity = [np.zeros_like(t) for t in d]
        self._velocity = [self.momentum * v + t for v, t in zip(self._velocity, d, strict=True)]
        return [self.learning_rate * v for v in self._velocity]

    def reset(self) -> None:
        self._velocity = None


class AdamServerOptimizer:
    """FedAdam, and FedYogi via ``yogi=True``. Algorithm 2 of Reddi et al.

    Per round, over the pseudo-gradient ``delta``:

        m <- beta1*m + (1-beta1)*delta
        v <- beta2*v + (1-beta2)*delta^2                       (Adam)
        v <- v - (1-beta2)*delta^2*sign(v - delta^2)           (Yogi)
        update = learning_rate * m / (sqrt(v) + tau)

    No bias correction, ``v`` starts at ``tau**2`` -- see the module
    docstring for why both are the paper's choices, not omissions.
    """

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
        yogi: bool = False,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError(f"server learning_rate must be > 0, got {learning_rate}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.tau = float(tau)
        self.yogi = bool(yogi)
        self.name = "fedyogi" if yogi else "fedadam"
        self._m: list[np.ndarray] | None = None
        self._v: list[np.ndarray] | None = None

    def step(self, delta: Weights) -> Weights:
        d = [np.asarray(t, dtype=np.float64) for t in delta]
        if self._m is None or self._v is None:
            self._m = [np.zeros_like(t) for t in d]
            self._v = [np.full_like(t, self.tau**2) for t in d]

        self._m = [self.beta1 * m + (1.0 - self.beta1) * t for m, t in zip(self._m, d, strict=True)]
        squares = [np.square(t) for t in d]
        if self.yogi:
            self._v = [
                v - (1.0 - self.beta2) * s * np.sign(v - s)
                for v, s in zip(self._v, squares, strict=True)
            ]
        else:
            self._v = [
                self.beta2 * v + (1.0 - self.beta2) * s
                for v, s in zip(self._v, squares, strict=True)
            ]
        return [
            self.learning_rate * m / (np.sqrt(v) + self.tau)
            for m, v in zip(self._m, self._v, strict=True)
        ]

    def reset(self) -> None:
        self._m = None
        self._v = None


def make_server_optimizer(
    name: str,
    *,
    learning_rate: float = 1.0,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.99,
    tau: float = 1e-3,
) -> ServerOptimizer:
    """Build a server optimizer by config name.

    Each name reads only its own hyperparameters: ``momentum`` is FedAvgM's,
    the betas and ``tau`` belong to FedAdam/FedYogi, and plain ``fedavg``
    reads nothing but the learning rate (whose non-1.0 values make it damped
    FedAvg -- still stateless, no longer the identity).
    """
    if name == "fedavg":
        return SGDServerOptimizer(learning_rate=learning_rate, momentum=0.0)
    if name == "fedavgm":
        return SGDServerOptimizer(learning_rate=learning_rate, momentum=momentum)
    if name in ("fedadam", "fedyogi"):
        return AdamServerOptimizer(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            tau=tau,
            yogi=(name == "fedyogi"),
        )
    raise ValueError(f"unknown server optimizer {name!r}; valid names: {VALID_SERVER_OPTIMIZERS}")
