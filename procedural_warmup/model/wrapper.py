"""ViT wrapper that bypasses the patch embedding during warm-up.

The wrapper feeds token ids through frozen token + positional embeddings, prepends the
backbone's CLS token, and runs the transformer blocks + final norm — never touching the
patch-embedding projection. Only the transformer blocks are trainable; those are exactly
the weights carried forward to image training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProceduralViT(nn.Module):
    def __init__(
        self,
        vit_backbone: nn.Module,
        tok_embed: nn.Module,
        pos_embed: nn.Module,
        H: int,
        W: int,
    ) -> None:
        super().__init__()
        self.vit = vit_backbone
        self.tok = tok_embed
        self.pos = pos_embed
        self.H, self.W = H, W
        self.N = H * W

    def _forward_core(self, x: torch.Tensor) -> torch.Tensor:
        """Run CLS-prepend -> transformer blocks -> final norm. ``x`` is ``(B, N, d)``."""
        B = x.shape[0]
        cls = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks:
            x = blk(x)
        return self.vit.norm(x)

    def forward_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Token ids ``(B, N)`` -> per-token features ``(B, N, d)`` (CLS dropped)."""
        B, N = token_ids.shape
        assert N == self.N, f"expected N={self.N} tokens, got {N}"
        x = self.tok(token_ids)
        pos_idx = torch.arange(N, device=token_ids.device).unsqueeze(0).expand(B, N)
        x = x + self.pos(pos_idx)
        return self._forward_core(x)[:, 1:, :]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(token_ids)
