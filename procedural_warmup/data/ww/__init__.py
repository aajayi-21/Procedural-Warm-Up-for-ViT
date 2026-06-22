"""WW source: a regular language (a random string concatenated with its exact copy).

In the paper WW is the negative control — simple repetition with no nested structure — and
it fails to help (it even hurts CIFAR-100). Included here so the method comparison is
complete. Masking targets the first half (predict the original from its copy).
"""

from __future__ import annotations

from procedural_warmup.data import register_source
from procedural_warmup.data.ww.dataset import WWGrid
from procedural_warmup.data.ww.masking import WWMasking


@register_source("ww")
def build_ww(cfg):
    return WWGrid(cfg), WWMasking(cfg)


__all__ = ["WWGrid", "WWMasking", "build_ww"]
