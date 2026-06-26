"""Build the warm-up model: a frozen-embedding ViT wrapper + an MLM prediction head."""

from __future__ import annotations

import timm
import torch.nn as nn

from procedural_warmup.model.embeddings import (
    Frozen1DRingPositionalEmbedding,
    Frozen2DSinCosPositionalEmbedding,
    FrozenPositionalEmbedding,
    FrozenTokenEmbedding,
)
from procedural_warmup.model.wrapper import ProceduralViT


def build_model(cfg) -> tuple[ProceduralViT, nn.Linear]:
    """Return ``(ProceduralViT, mlm_head)`` for the given warm-up config.

    The backbone is a timm ViT created with ``num_classes=0`` (no classifier); the
    masked-token head is a separate ``Linear(d, K)`` trained alongside and discarded
    before transfer.
    """
    backbone = timm.create_model(
        cfg.model.name,
        pretrained=False,
        num_classes=0,
        drop_path_rate=cfg.model.drop_path_rate,
    )
    embed_dim = backbone.embed_dim
    if embed_dim != cfg.model.embed_dim:
        raise ValueError(
            f"cfg.model.embed_dim={cfg.model.embed_dim} but backbone "
            f"'{cfg.model.name}' has embed_dim={embed_dim}"
        )

    N = cfg.grid.H * cfg.grid.W
    tok = FrozenTokenEmbedding(cfg.vocab.K, embed_dim)
    pos_kind = getattr(cfg.model, "pos_embed", "random")
    if pos_kind == "sincos2d":
        pos = Frozen2DSinCosPositionalEmbedding(N, embed_dim, cfg.grid.H, cfg.grid.W)
    elif pos_kind == "sincos1d":
        pos = Frozen1DRingPositionalEmbedding(N, embed_dim)
    elif pos_kind == "random":
        pos = FrozenPositionalEmbedding(N, embed_dim)
    else:
        raise ValueError(f"model.pos_embed must be random|sincos2d|sincos1d, got {pos_kind!r}")
    model = ProceduralViT(backbone, tok, pos, cfg.grid.H, cfg.grid.W)
    mlm_head = nn.Linear(embed_dim, cfg.vocab.K)
    return model, mlm_head
