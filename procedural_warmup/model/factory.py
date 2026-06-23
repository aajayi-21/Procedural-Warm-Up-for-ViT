"""Build the warm-up model: a frozen-embedding ViT wrapper + an MLM prediction head."""

from __future__ import annotations

import timm
import torch.nn as nn

from procedural_warmup.model.embeddings import (
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
    pos = FrozenPositionalEmbedding(N, embed_dim, trainable=not cfg.model.freeze_pos)
    model = ProceduralViT(backbone, tok, pos, cfg.grid.H, cfg.grid.W)
    mlm_head = nn.Linear(embed_dim, cfg.vocab.K)
    return model, mlm_head
