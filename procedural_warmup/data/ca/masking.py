"""Masked-prediction strategies for CA spacetime grids.

All modes operate on a batched ``(B, N)`` tensor (N = H*W, row-major = time-major) and
return the standard ``(masked_input, targets, mask)`` triple so the trainer is unchanged.

- ``random``  : MAE/BERT-style — each cell masked i.i.d. with prob ``mask_ratio``. Closest
                to the parent pipeline's objective; the default for the first run.
- ``forward`` : mask the last ``forward_rows`` time rows entirely — predict the future from
                the past. Deterministic forward evolution makes these targets cleanly
                well-posed (docs/cellular-automata.md, Stage 3).
- ``lightcone``: mask later-time cells with row-increasing probability, biasing the task
                toward cells whose value integrates a wider neighborhood (scaffold).
"""

from __future__ import annotations

import torch


class CAMasking:
    def __init__(self, cfg) -> None:
        self.H, self.W = cfg.grid.H, cfg.grid.W
        self.N = self.H * self.W
        self.mode = cfg.masking.mode
        self.mask_ratio = cfg.masking.mask_ratio
        self.forward_rows = cfg.masking.forward_rows
        self.mask_id = cfg.vocab.MASK_ID
        if self.mode == "forward" and not 0 < self.forward_rows < self.H:
            raise ValueError(
                f"forward_rows={self.forward_rows} must be in (0, H={self.H})"
            )

    def _mask_bool(self, batch_ids: torch.Tensor) -> torch.Tensor:
        B = batch_ids.shape[0]
        device = batch_ids.device
        if self.mode == "random":
            return torch.rand(B, self.N, device=device) < self.mask_ratio
        if self.mode == "forward":
            grid = torch.zeros(self.H, self.W, dtype=torch.bool, device=device)
            grid[self.H - self.forward_rows :, :] = True
            return grid.reshape(1, self.N).expand(B, self.N).clone()
        if self.mode == "lightcone":
            row_idx = torch.arange(self.H, device=device, dtype=torch.float)
            denom = max(self.H - 1, 1)
            p = ((row_idx / denom) * 2.0 * self.mask_ratio).clamp(0.0, 1.0)
            p = p.repeat_interleave(self.W).unsqueeze(0)  # (1, N)
            return torch.rand(B, self.N, device=device) < p
        raise ValueError(f"Unknown masking mode '{self.mode}'")

    def __call__(
        self, batch_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self._mask_bool(batch_ids)
        masked_input = batch_ids.clone()
        masked_input[mask] = self.mask_id
        return masked_input, batch_ids, mask
