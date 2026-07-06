"""Corner-close-only masking — the 2D analog of k-Dyck's close-only masking (H11).

Only d-corners (bottom-right of a matched rectangle) are eligible: predicting a masked
d requires binding to its visible a/b/c partners across both grid axes — long-range,
stack-like structure in 2D — and each masked target is uniquely determined by the
visible context (verified by :mod:`audit`; design principle P5). Eligibility (the
min-match-distance filter) is computed by the dataset and travels as channel 1 of the
``(B, 2, N)`` batch; masking never re-parses pictures.
"""

from __future__ import annotations

import torch

from procedural_warmup.data.dyck2d.alphabet import d_base


class CornerCloseOnlyMasking:
    def __init__(self, cfg) -> None:
        self.d_lo = d_base(cfg.dyck2d.k)
        self.d_hi = self.d_lo + cfg.dyck2d.k
        self.mask_ratio = cfg.masking.mask_ratio
        self.mask_id = cfg.vocab.MASK_ID

    def __call__(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch.ndim != 3 or batch.shape[1] != 2:
            # Fail loudly: a silent (B, N) fallback would run unfiltered masking if this
            # strategy were ever paired with a plain-ids dataset.
            raise ValueError(
                f"CornerCloseOnlyMasking expects (B, 2, N) two-channel batches "
                f"(ids, eligibility), got {tuple(batch.shape)}"
            )
        ids, elig = batch[:, 0], batch[:, 1].bool()
        # is_d is redundant with a correct eligibility channel but guards the shuffle
        # control (eligibility travels with the symbol) and future datasets.
        is_d = (ids >= self.d_lo) & (ids < self.d_hi)
        draw = torch.rand_like(ids, dtype=torch.float)
        mask = (draw < self.mask_ratio) & is_d & elig
        masked_input = ids.clone()
        masked_input[mask] = self.mask_id
        return masked_input, ids, mask
