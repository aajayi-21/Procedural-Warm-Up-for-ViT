"""Procedural warm-up for Vision Transformers — cellular-automata extension.

This package implements a lightweight procedural *warm-up* stage for ViTs: a brief
masked-token pretraining phase on procedurally generated, non-visual symbolic data
(formal grammars or cellular automata) whose learned structure transfers to standard
image classification.

Sub-packages
------------
- ``config``     : dataclass-based configuration + YAML loading.
- ``data``       : procedural data sources behind a common registry interface.
- ``model``      : frozen-embedding ViT wrapper used during warm-up.
- ``warmup``     : the masked-token pretraining stage and checkpoint stripping.
- ``downstream`` : standard image-classification training/transfer.
- ``analysis``   : reproducible figures, complexity metrics, layerwise probes.

See ``CLAUDE.md`` and ``docs/`` for the research context.
"""

__version__ = "0.1.0"
