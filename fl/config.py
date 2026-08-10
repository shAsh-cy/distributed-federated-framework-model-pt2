"""Typed configuration, loaded from YAML and validated on construction.

One object holds every tunable in the system. Nothing else in the codebase reads
environment variables or hard-codes a hyperparameter, so a run is fully described
by its config file plus its seed.

Validation is deliberately strict:

* Unknown keys are an error, not a warning. A silently ignored ``noise_multipler``
  typo would produce a run that claims differential privacy it does not have.
* Cross-field constraints are checked (see :meth:`Config._validate_cross_field`),
  because individually-valid values can still be jointly meaningless.

dataclasses rather than pydantic: TFF pins ``typing-extensions==4.5.*`` and
pydantic v2 requires ``>=4.6.1``, so pydantic cannot be installed alongside
tensorflow-federated. See requirements.txt.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PartitionScheme = str

VALID_PARTITIONS: tuple[str, ...] = ("iid", "dirichlet", "natural")
VALID_DATASETS: tuple[str, ...] = ("fashion_mnist", "femnist")
VALID_MODELS: tuple[str, ...] = ("small_cnn", "femnist_cnn")

#: The one model whose logits width matches each dataset's class count. A
#: 62-class model scoring 10-class data (or vice versa) fails loudly here
#: instead of silently training a mostly-dead logits layer.
DATASET_MODEL: dict[str, str] = {"fashion_mnist": "small_cnn", "femnist": "femnist_cnn"}


class ConfigError(ValueError):
    """Raised when a configuration file is structurally or semantically invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class DataConfig:
    """Dataset selection and how the training set is split across clients.

    For ``femnist`` the split is not synthesised: each client is one real
    writer from the LEAF-derived federated EMNIST, so ``partition`` must be
    ``"natural"`` and ``num_clients`` selects how many writers form the
    population (a seeded subsample of the 3,400 available; 3,400 selects all).
    For ``fashion_mnist`` the partition is synthetic, so ``"natural"`` is
    rejected — there are no real client boundaries to use.
    """

    dataset: str = "fashion_mnist"
    num_clients: int = 10
    partition: PartitionScheme = "dirichlet"
    dirichlet_alpha: float = 0.5

    def validate(self) -> None:
        _require(
            self.dataset in VALID_DATASETS,
            f"data.dataset must be one of {VALID_DATASETS}, got {self.dataset!r}",
        )
        _require(
            self.partition in VALID_PARTITIONS,
            f"data.partition must be one of {VALID_PARTITIONS}, got {self.partition!r}",
        )
        _require(self.num_clients >= 1, f"data.num_clients must be >= 1, got {self.num_clients}")
        _require(
            self.dirichlet_alpha > 0.0,
            f"data.dirichlet_alpha must be > 0, got {self.dirichlet_alpha}",
        )
        if self.dataset == "femnist":
            _require(
                self.partition == "natural",
                "data.dataset 'femnist' is partitioned by writer; set data.partition to "
                f"'natural' (got {self.partition!r}). Synthesising a split over naturally "
                "partitioned data would destroy the property the dataset exists to provide.",
            )
        else:
            _require(
                self.partition != "natural",
                f"data.partition 'natural' requires a dataset with real client boundaries "
                f"(femnist); {self.dataset!r} is a pooled dataset that must be split "
                "synthetically ('iid' or 'dirichlet')",
            )


@dataclass(frozen=True)
class ModelConfig:
    """Which architecture to federate."""

    name: str = "small_cnn"

    def validate(self) -> None:
        _require(
            self.name in VALID_MODELS,
            f"model.name must be one of {VALID_MODELS}, got {self.name!r}",
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Round schedule and the local optimiser each client runs."""

    rounds: int = 20
    client_fraction: float = 0.5
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    #: Where to write the final global model as an .npz checkpoint
    #: (fl/checkpoint.py). None disables the write on the gRPC and one-shot
    #: experiment paths; the coordinator always checkpoints per run id.
    checkpoint_path: str | None = None
    #: FedProx proximal coefficient mu (Li et al., MLSys 2020). When > 0 every
    #: client adds (mu/2)*||w - w_global||^2 to its local objective, which
    #: bounds how far local training can drift from the round's starting
    #: model. 0 disables the term: the local objective is exactly FedAvg's.
    #: Server-dictated per round (GetGlobalModelResponse.proximal_mu), like
    #: the other local hyperparameters.
    fedprox_mu: float = 0.0

    def validate(self) -> None:
        _require(self.rounds >= 1, f"training.rounds must be >= 1, got {self.rounds}")
        _require(
            self.fedprox_mu >= 0.0,
            f"training.fedprox_mu must be >= 0, got {self.fedprox_mu}",
        )
        _require(
            self.checkpoint_path is None or str(self.checkpoint_path).strip() != "",
            "training.checkpoint_path must be a non-empty path or omitted",
        )
        _require(
            0.0 < self.client_fraction <= 1.0,
            f"training.client_fraction must be in (0, 1], got {self.client_fraction}",
        )
        _require(
            self.local_epochs >= 1,
            f"training.local_epochs must be >= 1, got {self.local_epochs}",
        )
        _require(self.batch_size >= 1, f"training.batch_size must be >= 1, got {self.batch_size}")
        _require(
            self.learning_rate > 0.0,
            f"training.learning_rate must be > 0, got {self.learning_rate}",
        )
        _require(
            0.0 <= self.momentum < 1.0,
            f"training.momentum must be in [0, 1), got {self.momentum}",
        )


@dataclass(frozen=True)
class ServerConfig:
    """gRPC endpoint and the per-round barrier policy."""

    host: str = "0.0.0.0"
    port: int = 8080
    round_deadline_seconds: float = 120.0
    min_clients_per_round: int = 2
    registration_timeout_seconds: float = 300.0
    max_message_mb: int = 128

    def validate(self) -> None:
        _require(bool(self.host), "server.host must not be empty")
        _require(1 <= self.port <= 65535, f"server.port must be in [1, 65535], got {self.port}")
        _require(
            self.round_deadline_seconds > 0.0,
            f"server.round_deadline_seconds must be > 0, got {self.round_deadline_seconds}",
        )
        _require(
            self.min_clients_per_round >= 1,
            f"server.min_clients_per_round must be >= 1, got {self.min_clients_per_round}",
        )
        _require(
            self.registration_timeout_seconds > 0.0,
            "server.registration_timeout_seconds must be > 0, "
            f"got {self.registration_timeout_seconds}",
        )
        _require(
            self.max_message_mb >= 1,
            f"server.max_message_mb must be >= 1, got {self.max_message_mb}",
        )


@dataclass(frozen=True)
class ServerOptimizerConfig:
    """The FedOpt server optimizer (Reddi et al., ICLR 2021) applied each round.

    FedAvg adds the aggregated client delta to the global model directly; the
    FedOpt family treats that delta as a pseudo-gradient and lets a stateful
    server-side optimizer decide the step (:mod:`fl.server_optimizer`). The
    default -- ``fedavg`` at learning rate 1.0 -- is the exact identity and
    changes nothing about existing runs.

    Each name reads only its own fields: ``momentum`` belongs to ``fedavgm``;
    ``beta1``/``beta2``/``tau`` to ``fedadam`` and ``fedyogi``; ``fedavg``
    reads the learning rate alone. Unread fields are ignored, not rejected,
    so one config file can flip between optimizers by changing one line.

    Not composable with differential privacy (cross-field check below): the
    FedOpt path applies to the *weighted* mean, whose per-client sensitivity
    is unbounded. Post-processing the noised uniform mean would be sound DP,
    but this repo has not wired or measured that composition, and running an
    unmeasured mechanism behind a config flag is how accounting fictions ship.
    """

    name: str = "fedavg"
    learning_rate: float = 1.0
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.99
    tau: float = 1e-3

    def validate(self) -> None:
        # Lazy import: fl.server_optimizer pulls numpy, and this module stays
        # importable without it until a config is actually constructed.
        from .server_optimizer import VALID_SERVER_OPTIMIZERS

        _require(
            self.name in VALID_SERVER_OPTIMIZERS,
            f"server_optimizer.name must be one of {VALID_SERVER_OPTIMIZERS}, got {self.name!r}",
        )
        _require(
            self.learning_rate > 0.0,
            f"server_optimizer.learning_rate must be > 0, got {self.learning_rate}",
        )
        _require(
            0.0 <= self.momentum < 1.0,
            f"server_optimizer.momentum must be in [0, 1), got {self.momentum}",
        )
        _require(
            0.0 <= self.beta1 < 1.0,
            f"server_optimizer.beta1 must be in [0, 1), got {self.beta1}",
        )
        _require(
            0.0 <= self.beta2 < 1.0,
            f"server_optimizer.beta2 must be in [0, 1), got {self.beta2}",
        )
        _require(self.tau > 0.0, f"server_optimizer.tau must be > 0, got {self.tau}")


@dataclass(frozen=True)
class PrivacyConfig:
    """Client-level differential privacy applied at the aggregation step.

    The granularity here is **client-level** (also called user-level): the unit of
    protection is one participant's entire local dataset, so the guarantee is that
    the released global model is statistically near-indistinguishable whether or
    not any single client took part.

    This is *not* example-level DP. Example-level DP (what DP-SGD inside a single
    trainer gives you) protects one training example and says nothing about
    whether a given participant contributed at all. Conflating the two overstates
    the guarantee, which is why the distinction is spelled out here and enforced
    in :mod:`fl.aggregation`: ``l2_clip_norm`` bounds one *client's* whole update.

    ``epsilon`` is absent from this object by design. Epsilon is a computed
    consequence of the noise multiplier, the client sampling rate and the number
    of rounds -- see :func:`fl.aggregation.compute_epsilon`. It is never a knob.
    """

    enabled: bool = False
    l2_clip_norm: float = 1.0
    noise_multiplier: float = 0.0
    delta: float = 1e-5
    #: Quantile-based adaptive clipping (Andrew et al. 2021, via TFF's
    #: gaussian_adaptive factory). When true, ``l2_clip_norm`` is the INITIAL
    #: clip estimate and the clip then tracks the configured quantile of the
    #: actual update norms. The fixed-norm path remains the default.
    adaptive_clipping: bool = False
    #: Fraction of updates that should escape clipping; 0.5 tracks the median.
    adaptive_target_quantile: float = 0.5
    #: Geometric adaptation rate: the clip moves by at most exp(rate)/round.
    adaptive_learning_rate: float = 0.2
    #: Stddev of the noise on the clipped-count used for the quantile estimate.
    #: None uses TFF's default (clients_per_round / 20). Part of the privacy
    #: budget -- see fl.aggregation.adaptive_noise_breakdown.
    adaptive_clipped_count_stddev: float | None = None

    def validate(self) -> None:
        _require(
            self.l2_clip_norm > 0.0,
            f"privacy.l2_clip_norm must be > 0, got {self.l2_clip_norm}",
        )
        _require(
            self.noise_multiplier >= 0.0,
            f"privacy.noise_multiplier must be >= 0, got {self.noise_multiplier}",
        )
        _require(0.0 < self.delta < 1.0, f"privacy.delta must be in (0, 1), got {self.delta}")
        _require(
            0.0 < self.adaptive_target_quantile < 1.0,
            "privacy.adaptive_target_quantile must be strictly inside (0, 1) -- 0 would "
            "clip everything and 1 nothing, and neither is a quantile to track; got "
            f"{self.adaptive_target_quantile}",
        )
        _require(
            self.adaptive_learning_rate > 0.0,
            f"privacy.adaptive_learning_rate must be > 0, got {self.adaptive_learning_rate}",
        )
        if self.adaptive_clipped_count_stddev is not None:
            _require(
                self.adaptive_clipped_count_stddev > 0.0,
                "privacy.adaptive_clipped_count_stddev must be > 0 when set, got "
                f"{self.adaptive_clipped_count_stddev}",
            )
        if self.enabled:
            _require(
                self.noise_multiplier > 0.0,
                "privacy.enabled is true but privacy.noise_multiplier is 0; that provides "
                "no privacy at all (epsilon would be infinite). Set a positive multiplier "
                "or set privacy.enabled to false.",
            )


@dataclass(frozen=True)
class Config:
    """Root configuration object."""

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    server_optimizer: ServerOptimizerConfig = field(default_factory=ServerOptimizerConfig)

    _SECTIONS = {
        "data": DataConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "server": ServerConfig,
        "privacy": PrivacyConfig,
        "server_optimizer": ServerOptimizerConfig,
    }

    def __post_init__(self) -> None:
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.server.validate()
        self.privacy.validate()
        self.server_optimizer.validate()
        self._validate_cross_field()

    def _validate_cross_field(self) -> None:
        """Constraints that no single section can check on its own."""
        _require(
            DATASET_MODEL[self.data.dataset] == self.model.name,
            f"model.name {self.model.name!r} does not match data.dataset "
            f"{self.data.dataset!r}; expected {DATASET_MODEL[self.data.dataset]!r} "
            "(the logits width must equal the dataset's class count)",
        )
        _require(
            self.server.min_clients_per_round <= self.data.num_clients,
            f"server.min_clients_per_round ({self.server.min_clients_per_round}) exceeds "
            f"data.num_clients ({self.data.num_clients}); no round could ever reach quorum",
        )
        _require(
            self.clients_per_round >= self.server.min_clients_per_round,
            f"training.client_fraction ({self.training.client_fraction}) samples only "
            f"{self.clients_per_round} of {self.data.num_clients} clients, which is below "
            f"server.min_clients_per_round ({self.server.min_clients_per_round}); "
            "every round would fail quorum",
        )
        _require(
            not (
                self.privacy.enabled
                and (
                    self.server_optimizer.name != "fedavg"
                    or self.server_optimizer.learning_rate != 1.0
                )
            ),
            f"server_optimizer {self.server_optimizer.name!r} "
            f"(learning_rate={self.server_optimizer.learning_rate}) cannot be combined "
            "with privacy.enabled: the FedOpt path applies to the weighted mean, whose "
            "per-client sensitivity is unbounded, so the DP accounting would not hold. "
            "Run FedOpt without DP, or DP with the default fedavg server step.",
        )

    @property
    def clients_per_round(self) -> int:
        """Number of clients sampled per round: ``max(1, ceil(C * N))``."""
        import math

        return max(1, math.ceil(self.training.client_fraction * self.data.num_clients))

    @property
    def client_sampling_rate(self) -> float:
        """Client sampling rate ``q`` used by the privacy accountant."""
        return self.clients_per_round / self.data.num_clients

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        """Build a Config from a plain dict, rejecting unknown keys."""
        if not isinstance(raw, dict):
            raise ConfigError(f"configuration root must be a mapping, got {type(raw).__name__}")

        known_top = {"seed", *cls._SECTIONS}
        unknown_top = set(raw) - known_top
        if unknown_top:
            raise ConfigError(
                f"unknown top-level configuration key(s): {sorted(unknown_top)}. "
                f"Valid keys are {sorted(known_top)}"
            )

        kwargs: dict[str, Any] = {}
        if "seed" in raw:
            kwargs["seed"] = raw["seed"]

        for name, section_cls in cls._SECTIONS.items():
            if name not in raw:
                continue
            section = raw[name]
            if section is None:
                continue
            if not isinstance(section, dict):
                raise ConfigError(
                    f"configuration section {name!r} must be a mapping, "
                    f"got {type(section).__name__}"
                )
            valid = {f.name for f in dataclasses.fields(section_cls)}
            unknown = set(section) - valid
            if unknown:
                raise ConfigError(
                    f"unknown key(s) in section {name!r}: {sorted(unknown)}. "
                    f"Valid keys are {sorted(valid)}"
                )
            kwargs[name] = section_cls(**section)

        seed = kwargs.get("seed", 42)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ConfigError(f"seed must be an integer, got {seed!r}")

        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load and validate a configuration file."""
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"configuration file not found: {p}")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"{p} is not valid YAML: {exc}") from exc
        if raw is None:
            raw = {}
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable plain-dict view, suitable for JSON metrics output."""
        return {
            "seed": self.seed,
            **{name: dataclasses.asdict(getattr(self, name)) for name in self._SECTIONS},
        }

    def replace(self, **section_overrides: dict[str, Any]) -> Config:
        """Return a new Config with per-section field overrides applied.

        ``cfg.replace(privacy={"enabled": True, "noise_multiplier": 1.0})``
        """
        raw = self.to_dict()
        for section, overrides in section_overrides.items():
            if section not in self._SECTIONS:
                raise ConfigError(f"unknown section {section!r}")
            raw[section].update(overrides)
        return Config.from_dict(raw)
