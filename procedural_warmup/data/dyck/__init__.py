"""k-Dyck procedural source (reference-parity baseline)."""

from __future__ import annotations

from procedural_warmup.data import register_source
from procedural_warmup.data.dyck.dataset import DyckGrid
from procedural_warmup.data.dyck.masking import CloseOnlyMasking


@register_source("dyck")
def build_dyck(cfg):
    """Return ``(DyckGrid, CloseOnlyMasking)`` for the current config."""
    return DyckGrid(cfg), CloseOnlyMasking(cfg)


__all__ = ["DyckGrid", "CloseOnlyMasking", "build_dyck"]
