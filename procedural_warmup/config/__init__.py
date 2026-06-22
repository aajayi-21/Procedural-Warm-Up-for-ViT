"""Configuration system (dataclasses + YAML overrides)."""

from procedural_warmup.config.schema import (
    RootConfig,
    DownstreamConfig,
    load_config,
    load_downstream_config,
    config_to_dict,
)

__all__ = [
    "RootConfig",
    "DownstreamConfig",
    "load_config",
    "load_downstream_config",
    "config_to_dict",
]
