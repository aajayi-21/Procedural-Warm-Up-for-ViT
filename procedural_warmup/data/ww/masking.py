"""First-half masking for WW (predict the original substring from its copy).

The "contextually informed" tokens for WW are the first half of the sequence; each is
masked with probability ``mask_ratio`` and must be recovered from the visible copy in the
second half. Returns the standard ``(masked_input, targets, mask)`` triple.
"""

from __future__ import annotations

import torch


class WWMasking:
    def __init__(self, cfg) -> None:
        self.N = cfg.grid.H * cfg.grid.W
        self.half = self.N // 2
        self.mask_ratio = cfg.masking.mask_ratio
        self.mask_id = cfg.vocab.MASK_ID

    def __call__(
        self, batch_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = batch_ids.shape
        eligible = torch.zeros(N, dtype=torch.bool, device=batch_ids.device)
        eligible[: self.half] = True  # first half = the tokens to predict from the copy
        draw = torch.rand(B, N, device=batch_ids.device)
        mask = (draw < self.mask_ratio) & eligible.unsqueeze(0)
        masked_input = batch_ids.clone()
        masked_input[mask] = self.mask_id
        return masked_input, batch_ids, mask
