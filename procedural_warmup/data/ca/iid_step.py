"""IID single-step CA transduction — Phase-1 texture-vs-operator test.

These sources isolate *learning the local CA update operator* from *reconstructing CA
texture*. Instead of an evolved spacetime trajectory (which develops recognizable
textures the model could memorize), each sample is a single deterministic rule step on an
i.i.d. random board::

    x ~ Bernoulli(p)        # structureless input — no evolved texture
    y = ECA_rule(x)         # one local-operator step

The model sees the *full* current state ``x`` and must predict the *full* next state ``y``;
no target token is ever fed to it (strict full-mask transduction). If this warm-up transfers
to images, the model is learning a transition operator, not CA texture.

Two controls share the exact same geometry / tokenization / target marginals:

- ``mode="shuffled"`` (:class:`CaStepDataset`): target is ``rule(z)`` for an *unrelated*
  ``z ~ Bernoulli(p)`` at the same density. Identical output distribution, no causal link to
  the input. **``true`` beating ``shuffled`` downstream is the operator-learning signal.**
- :class:`IidBoardDataset` + random masking: inpaint an i.i.d. board — the structure-free
  reconstruction floor (unlearnable beyond the per-cell marginal).

Density is sampled per example from ``cfg.ca_step.densities`` so the target marginal is not a
single fixed bias to exploit; ``shuffled`` draws ``z`` from the same density bucket as ``x``.
The board is a 1-D ring of ``N = H*W`` cells with binary tokenization (ids {2, 3}), matching
the ECA rule used by the Stage-1 CA experiments.
"""

from __future__ import annotations

import numpy as np
import torch

from procedural_warmup.data.base import ProceduralDataset
from procedural_warmup.data.ca import tokenize as tok
from procedural_warmup.data.ca.eca import rule_table, step


class CaStepDataset(ProceduralDataset):
    """One deterministic ECA step on an i.i.d. board; yields ``[x | target]`` (length ``2N``).

    ``target = rule(x)`` for ``mode="true"`` or ``rule(z)`` (independent ``z`` at the same
    density) for ``mode="shuffled"``. The split into input/target is done by
    :class:`TransductionMasking`, which stays rule-agnostic.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.N = cfg.grid.H * cfg.grid.W
        self.table = rule_table(cfg.ca_step.rule)
        self.boundary = cfg.ca_step.boundary
        self.densities = list(cfg.ca_step.densities)
        # YAML coerces bare ``true`` to a bool; normalize so 'true'/True both work.
        self.mode = str(cfg.ca_step.mode).lower()
        if self.mode not in ("true", "shuffled"):
            raise ValueError(f"ca_step.mode must be 'true'|'shuffled', got {self.mode!r}")
        if cfg.vocab.K < tok.vocab_size_binary():
            raise ValueError(f"vocab.K={cfg.vocab.K} < {tok.vocab_size_binary()} for binary CA")

    def __len__(self) -> int:
        return self.cfg.dataset.n_samples

    def __getitem__(self, _idx: int) -> torch.LongTensor:
        rng = np.random.default_rng()
        p = float(rng.choice(self.densities))
        x = (rng.random(self.N) < p).astype(np.uint8)
        # "true": evolve the input itself; "shuffled": evolve an unrelated board at same p.
        src = x if self.mode == "true" else (rng.random(self.N) < p).astype(np.uint8)
        y = step(src, self.table, self.boundary)
        pair = np.concatenate([tok.binary_tokens(x), tok.binary_tokens(y)])  # (2N,)
        return torch.tensor(pair, dtype=torch.long)


class IidBoardDataset(ProceduralDataset):
    """A single i.i.d. Bernoulli board (length ``N``) — the structure-free reconstruction floor.

    Paired with the standard random masking, inpainting an i.i.d. board is unlearnable beyond
    the per-cell marginal, so any downstream benefit cannot come from CA structure.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.N = cfg.grid.H * cfg.grid.W
        self.densities = list(cfg.ca_step.densities)
        if cfg.vocab.K < tok.vocab_size_binary():
            raise ValueError(f"vocab.K={cfg.vocab.K} < {tok.vocab_size_binary()} for binary CA")

    def __len__(self) -> int:
        return self.cfg.dataset.n_samples

    def __getitem__(self, _idx: int) -> torch.LongTensor:
        rng = np.random.default_rng()
        p = float(rng.choice(self.densities))
        x = (rng.random(self.N) < p).astype(np.uint8)
        return torch.tensor(tok.binary_tokens(x), dtype=torch.long)


class TransductionMasking:
    """Full-state -> full-state target. Input is ``x`` (all visible); predict *all* of ``y``.

    Consumes the ``[x | y]`` pair from :class:`CaStepDataset` (shape ``(B, 2N)``) and returns
    ``(x, y, ones)`` so the trainer takes loss at every position with **no** target token fed
    to the model — a strict full mask. ``mask_ratio``/``masking.mode`` are ignored here.
    """

    def __init__(self, cfg) -> None:
        self.N = cfg.grid.H * cfg.grid.W

    def __call__(self, batch_ids: torch.Tensor):
        x = batch_ids[:, : self.N].contiguous()
        y = batch_ids[:, self.N :].contiguous()
        mask = torch.ones_like(x, dtype=torch.bool)
        return x, y, mask
