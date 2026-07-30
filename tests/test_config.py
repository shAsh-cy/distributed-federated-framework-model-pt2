"""Tests for the typed configuration object."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from fl.config import Config, ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def test_default_yaml_loads_and_validates():
    cfg = Config.from_yaml(CONFIG_DIR / "default.yaml")
    assert cfg.seed == 42
    assert cfg.data.num_clients == 10
    assert cfg.data.partition == "dirichlet"
    assert cfg.privacy.enabled is False


@pytest.mark.parametrize("name", ["default.yaml", "dp_moderate.yaml", "dp_high.yaml"])
def test_every_shipped_config_is_valid(name):
    """Every config committed to the repo must load. Catches drift in the examples."""
    path = CONFIG_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not present yet")
    Config.from_yaml(path)


def test_defaults_are_non_iid():
    """The default must be non-IID; an IID default would hide the hard part."""
    cfg = Config.from_yaml(CONFIG_DIR / "default.yaml")
    assert cfg.data.partition == "dirichlet"


def test_clients_per_round_rounds_up():
    cfg = Config.from_dict({"data": {"num_clients": 10}, "training": {"client_fraction": 0.25}})
    assert cfg.clients_per_round == math.ceil(0.25 * 10) == 3
    assert cfg.client_sampling_rate == pytest.approx(0.3)


def test_clients_per_round_is_at_least_one():
    cfg = Config.from_dict(
        {
            "data": {"num_clients": 3},
            "training": {"client_fraction": 0.01},
            "server": {"min_clients_per_round": 1},
        }
    )
    assert cfg.clients_per_round == 1


def test_unknown_top_level_key_rejected():
    with pytest.raises(ConfigError, match="unknown top-level configuration key"):
        Config.from_dict({"trainng": {}})


def test_unknown_section_key_rejected():
    """A typo'd privacy knob must fail loudly rather than silently do nothing."""
    with pytest.raises(ConfigError, match="noise_multipler"):
        Config.from_dict({"privacy": {"enabled": True, "noise_multipler": 1.0}})


def test_section_must_be_mapping():
    with pytest.raises(ConfigError, match="must be a mapping"):
        Config.from_dict({"training": [1, 2, 3]})


def test_root_must_be_mapping():
    with pytest.raises(ConfigError, match="root must be a mapping"):
        Config.from_dict([1, 2, 3])


@pytest.mark.parametrize(
    ("section", "payload", "message"),
    [
        ("training", {"client_fraction": 0.0}, "client_fraction"),
        ("training", {"client_fraction": 1.5}, "client_fraction"),
        ("training", {"rounds": 0}, "rounds"),
        ("training", {"local_epochs": 0}, "local_epochs"),
        ("training", {"batch_size": 0}, "batch_size"),
        ("training", {"learning_rate": 0.0}, "learning_rate"),
        ("training", {"momentum": 1.0}, "momentum"),
        ("data", {"num_clients": 0}, "num_clients"),
        ("data", {"dirichlet_alpha": 0.0}, "dirichlet_alpha"),
        ("data", {"partition": "shuffled"}, "partition"),
        ("data", {"dataset": "cifar100"}, "dataset"),
        ("model", {"name": "resnet50"}, "model.name"),
        ("server", {"port": 0}, "port"),
        ("server", {"port": 70000}, "port"),
        ("server", {"round_deadline_seconds": 0.0}, "round_deadline_seconds"),
        ("server", {"max_message_mb": 0}, "max_message_mb"),
        ("privacy", {"l2_clip_norm": 0.0}, "l2_clip_norm"),
        ("privacy", {"noise_multiplier": -1.0}, "noise_multiplier"),
        ("privacy", {"delta": 0.0}, "delta"),
        ("privacy", {"delta": 1.0}, "delta"),
    ],
)
def test_out_of_range_values_rejected(section, payload, message):
    with pytest.raises(ConfigError, match=message):
        Config.from_dict({section: payload})


def test_dp_enabled_with_zero_noise_is_rejected():
    """Claiming DP while adding no noise is the single most misleading misconfiguration."""
    with pytest.raises(ConfigError, match="no privacy at all"):
        Config.from_dict({"privacy": {"enabled": True, "noise_multiplier": 0.0}})


def test_quorum_above_client_count_rejected():
    with pytest.raises(ConfigError, match="no round could ever reach quorum"):
        Config.from_dict(
            {"data": {"num_clients": 2}, "server": {"min_clients_per_round": 5}}
        )


def test_sampled_clients_below_quorum_rejected():
    """C * N must reach the quorum, or every round fails."""
    with pytest.raises(ConfigError, match="below server.min_clients_per_round"):
        Config.from_dict(
            {
                "data": {"num_clients": 10},
                "training": {"client_fraction": 0.1},
                "server": {"min_clients_per_round": 5},
            }
        )


def test_seed_must_be_int():
    with pytest.raises(ConfigError, match="seed must be an integer"):
        Config.from_dict({"seed": "42"})


def test_bool_is_not_accepted_as_seed():
    with pytest.raises(ConfigError, match="seed must be an integer"):
        Config.from_dict({"seed": True})


def test_missing_file_raises_config_error():
    with pytest.raises(ConfigError, match="configuration file not found"):
        Config.from_yaml(CONFIG_DIR / "does_not_exist.yaml")


def test_invalid_yaml_raises_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("data: {unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        Config.from_yaml(bad)


def test_empty_yaml_yields_defaults(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert Config.from_yaml(empty) == Config()


def test_to_dict_round_trips():
    cfg = Config.from_yaml(CONFIG_DIR / "default.yaml")
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_replace_overrides_a_section():
    cfg = Config.from_yaml(CONFIG_DIR / "default.yaml")
    dp = cfg.replace(privacy={"enabled": True, "noise_multiplier": 1.0})
    assert dp.privacy.enabled is True
    assert dp.privacy.noise_multiplier == 1.0
    # Original is untouched: the config is frozen.
    assert cfg.privacy.enabled is False


def test_replace_revalidates():
    cfg = Config.from_yaml(CONFIG_DIR / "default.yaml")
    with pytest.raises(ConfigError, match="no privacy at all"):
        cfg.replace(privacy={"enabled": True})


def test_replace_unknown_section_rejected():
    cfg = Config()
    with pytest.raises(ConfigError, match="unknown section"):
        cfg.replace(nonsense={"a": 1})


def test_config_is_frozen():
    cfg = Config()
    with pytest.raises(Exception):
        cfg.seed = 1
