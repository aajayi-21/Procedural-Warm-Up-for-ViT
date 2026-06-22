"""Close-only masking for k-Dyck (the reference's masked-token objective).

Only *closing* brackets are eligible to be masked: predicting them requires the model to
track the nesting state, which is exactly the structural dependency the warm-up should
instil. Operates on a batched ``(B, N)`` tensor and returns the standard triple.
"""

from __future__ import annotations

import torch


class CloseOnlyMasking:
    def __init__(self, cfg) -> None:
        self.close_base = 2 + cfg.dyck.k_open
        self.k_close = cfg.dyck.k_close
        self.mask_ratio = cfg.masking.mask_ratio
        self.mask_id = cfg.vocab.MASK_ID

    def __call__(
        self, batch_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_close = (batch_ids >= self.close_base) & (
            batch_ids < self.close_base + self.k_close
        )
        draw = torch.rand_like(batch_ids, dtype=torch.float)
        mask = (draw < self.mask_ratio) & is_close
        masked_input = batch_ids.clone()
        masked_input[mask] = self.mask_id
        return masked_input, batch_ids, mask
