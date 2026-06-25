"""k-Dyck procedural source (reference-parity baseline)."""

from __future__ import annotations

from procedural_warmup.data import register_source
from procedural_warmup.data.dyck.dataset import DyckGrid
from procedural_warmup.data.dyck.masking import CloseOnlyMasking
from procedural_warmup.data.dyck.spatial import SpatialDyckGrid


@register_source("dyck")
def build_dyck(cfg):
    """Return ``(DyckGrid, CloseOnlyMasking)`` for the current config."""
    return DyckGrid(cfg), CloseOnlyMasking(cfg)


@register_source("spatial_dyck")
def build_spatial_dyck(cfg):
    """Same k-Dyck tree, 1-D / nested-2-D / permuted layout (``cfg.spatial_dyck.mode``).

    Reuses CloseOnlyMasking unchanged so *only* the token layout differs from ``dyck``.
    """
    return SpatialDyckGrid(cfg), CloseOnlyMasking(cfg)


__all__ = [
    "DyckGrid",
    "SpatialDyckGrid",
    "CloseOnlyMasking",
    "build_dyck",
    "build_spatial_dyck",
]
