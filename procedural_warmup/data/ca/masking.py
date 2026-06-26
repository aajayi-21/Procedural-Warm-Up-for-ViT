"""Masked-prediction strategies for CA spacetime grids.

All modes operate on a batched ``(B, N)`` tensor (N = H*W, row-major = time-major) and
return the standard ``(masked_input, targets, mask)`` triple so the trainer is unchanged.

- ``random``  : MAE/BERT-style — each cell masked i.i.d. with prob ``mask_ratio``. Closest
                to the parent pipeline's objective; the default for the first run.
- ``forward`` : mask the last ``forward_rows`` time rows entirely — predict the future from
                the past. ``forward_rows=1`` is the single-step "predict the entire next state
                vector from the past" objective. NOTE: these targets are *deterministically
                well-posed only when the spacetime is a self-contained trajectory*, i.e. when
                ``ca.sim_width == cell_W`` (no crop, so the whole grid is one closed strip). With
                a wider cropped ``sim_width`` the deepest masked rows depend on out-of-window
                history and become partly aleatoric — use ``sim_width: 14`` (binary) for clean
                forward/next-state runs (docs/cellular-automata.md, Stage 3).
- ``lightcone``: mask later-time cells with row-increasing probability, biasing the task
                toward cells whose value integrates a wider neighborhood (scaffold).
- ``block2d`` : mask contiguous ``block_h x block_w`` rectangles of the *2-D* (time x space)
                grid (MAE-style block masking) instead of i.i.d. cells. A masked cell's
                immediate neighbours are usually masked too, so it cannot be copied from an
                adjacent visible cell — reconstruction needs genuine 2-D integration over the
                rule. Pairs with the ``sincos2d`` positional embedding to retain 2-D structure
                (H3 test). Because the stamped rectangles overlap, realized coverage is somewhat
                *below* ``mask_ratio`` (~0.45 at ratio 0.5, 4x4 blocks, 14x14 grid); the block
                count is set by inclusion-exclusion to approach the target.
"""

from __future__ import annotations

import math

import torch


class CAMasking:
    def __init__(self, cfg) -> None:
        self.H, self.W = cfg.grid.H, cfg.grid.W
        self.N = self.H * self.W
        self.mode = cfg.masking.mode
        self.mask_ratio = cfg.masking.mask_ratio
        self.forward_rows = cfg.masking.forward_rows
        self.block_h = cfg.masking.block_h
        self.block_w = cfg.masking.block_w
        self.mask_id = cfg.vocab.MASK_ID
        if self.mode == "forward" and not 0 < self.forward_rows < self.H:
            raise ValueError(
                f"forward_rows={self.forward_rows} must be in (0, H={self.H})"
            )
        if self.mode == "block2d" and not (
            0 < self.block_h <= self.H and 0 < self.block_w <= self.W
        ):
            raise ValueError(
                f"block2d needs 0<block_h<=H and 0<block_w<=W, got "
                f"block_h={self.block_h}, block_w={self.block_w} (H={self.H}, W={self.W})"
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
        if self.mode == "block2d":
            # Stamp ``num_blocks`` random block_h x block_w rectangles per sample (overlaps
            # collapse). Block count from inclusion-exclusion so expected union ~ mask_ratio:
            # E[coverage] = 1 - (1 - block_area/N)^num_blocks. Fully vectorized (no host<->device
            # sync), so it stays on the CUDA hot path.
            block_area = self.block_h * self.block_w
            if block_area >= self.N:
                num_blocks = 1
            else:
                frac = min(self.mask_ratio, 0.999)
                num_blocks = max(
                    1, math.ceil(math.log(1.0 - frac) / math.log(1.0 - block_area / self.N))
                )
            tops = torch.randint(0, self.H - self.block_h + 1, (B, num_blocks), device=device)
            lefts = torch.randint(0, self.W - self.block_w + 1, (B, num_blocks), device=device)
            rows = torch.arange(self.H, device=device)
            cols = torch.arange(self.W, device=device)
            row_in = (rows[None, None, :] >= tops[:, :, None]) & (
                rows[None, None, :] < tops[:, :, None] + self.block_h
            )  # (B, num_blocks, H)
            col_in = (cols[None, None, :] >= lefts[:, :, None]) & (
                cols[None, None, :] < lefts[:, :, None] + self.block_w
            )  # (B, num_blocks, W)
            grid = (row_in[:, :, :, None] & col_in[:, :, None, :]).any(dim=1)  # (B, H, W)
            return grid.reshape(B, self.N)
        raise ValueError(f"Unknown masking mode '{self.mode}'")

    def __call__(
        self, batch_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self._mask_bool(batch_ids)
        masked_input = batch_ids.clone()
        masked_input[mask] = self.mask_id
        return masked_input, batch_ids, mask
