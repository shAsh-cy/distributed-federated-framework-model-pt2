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

VALID_PARTITIONS: tuple[str, ...] = ("iid", "dirichlet")
VALID_DATASETS: tuple[str, ...] = ("fashion_mnist",)
VALID_MODELS: tuple[str, ...] = ("small_cnn",)


class ConfigError(ValueError):
    """Raised when a configuration file is structurally or semantically invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class DataConfig:
    """Dataset selection and how the training set is split across clients."""

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

    def validate(self) -> None:
        _require(self.rounds >= 1, f"training.rounds must be >= 1, got {self.rounds}")
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

    _SECTIONS = {
        "data": DataConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "server": ServerConfig,
        "privacy": PrivacyConfig,
    }

    def __post_init__(self) -> None:
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.server.validate()
        self.privacy.validate()
        self._validate_cross_field()

    def _validate_cross_field(self) -> None:
        """Constraints that no single section can check on its own."""
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
