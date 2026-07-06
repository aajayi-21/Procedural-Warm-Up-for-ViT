"""Procedural data-source registry.

The warm-up CLI/trainer obtains its data via :func:`build_source`, never by branching on
the source name. New sources self-register with :func:`register_source` at import time, so
plugging one in requires no change to the core loop.

    @register_source("ca")
    def build_ca(cfg) -> tuple[ProceduralDataset, MaskingStrategy]:
        ...
"""

from __future__ import annotations

from typing import Callable

from procedural_warmup.data.base import MaskingStrategy, ProceduralDataset

# name -> builder(cfg) -> (dataset, masking)
_REGISTRY: dict[str, Callable[..., tuple[ProceduralDataset, MaskingStrategy]]] = {}


def register_source(name: str):
    """Decorator registering a source builder under ``name``."""

    def _wrap(builder):
        if name in _REGISTRY:
            raise ValueError(f"Data source '{name}' is already registered")
        _REGISTRY[name] = builder
        return builder

    return _wrap


def build_source(cfg) -> tuple[ProceduralDataset, MaskingStrategy]:
    """Instantiate the ``(dataset, masking)`` pair selected by ``cfg.data.source``."""
    name = cfg.data.source
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown data source '{name}'. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](cfg)


def available_sources() -> list[str]:
    return sorted(_REGISTRY)


# Import source packages so their @register_source builders run. Keep these at the bottom
# to avoid circular imports (the submodules import from this module).
from procedural_warmup.data import dyck as _dyck  # noqa: E402,F401
from procedural_warmup.data import dyck2d as _dyck2d  # noqa: E402,F401
from procedural_warmup.data import dyck_shuffle as _dyck_shuffle  # noqa: E402,F401
from procedural_warmup.data import ww as _ww  # noqa: E402,F401
from procedural_warmup.data import ca as _ca  # noqa: E402,F401

__all__ = [
    "ProceduralDataset",
    "MaskingStrategy",
    "register_source",
    "build_source",
    "available_sources",
]
