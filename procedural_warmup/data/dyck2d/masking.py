"""Closing-corner masking — the 2D analog of k-Dyck's close-only masking (H11).

Two policies, selected by ``cfg.dyck2d.mask_roles``:

- ``"d"`` (corner-close-only, the H6 default): only d-corners (bottom-right, the
  doubly-closing role) are eligible. Each masked d is uniquely forced by its visible
  a/b/c partners via EITHER its row or its column — the information is redundant, so
  single-axis parsing suffices (measured consequence: the task saturates early at
  14x14, see results/reports/dw32-screen/report.md §5).
- ``"cd"`` (both closing roles, the harder H11-style objective): c-corners
  (bottom-left; row-openers but column-closers) are eligible too. A masked c has NO
  row witness — only its column (the visible a above) determines it — and a masked d
  whose row-mate c is also hidden is likewise forced only by its column (the visible
  b). Single-axis shortcuts stop sufficing, supervision density roughly doubles, and
  every target is still uniquely determined under the policy (a/b never masked; the
  ``closing`` audit mode proves it and brute force cross-checks it in tests).

Eligibility (which corners pass the min-match-distance filter, under
``cfg.dyck2d.filter_mode``) is computed by the dataset and travels as channel 1 of the
``(B, 2, N)`` batch; masking never re-parses pictures. Thanks to the role-major id
layout the eligible-role test is a single contiguous range comparison: the c-block
[2+2k, 2+3k) and d-block [2+3k, 2+4k) are adjacent.
"""

from __future__ import annotations

import torch

from procedural_warmup.data.dyck2d.alphabet import N_SPECIAL, d_base


class CornerCloseOnlyMasking:
    def __init__(self, cfg) -> None:
        roles = cfg.dyck2d.mask_roles
        if roles == "d":
            self.lo = d_base(cfg.dyck2d.k)
        elif roles == "cd":
            self.lo = N_SPECIAL + 2 * cfg.dyck2d.k  # c-block start
        else:
            raise ValueError(f"dyck2d.mask_roles must be 'd'|'cd', got {roles!r}")
        self.hi = N_SPECIAL + 4 * cfg.dyck2d.k  # end of the d-block
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
        # The role test is redundant with a correct eligibility channel but guards the
        # shuffle control (eligibility travels with the symbol) and future datasets.
        in_role = (ids >= self.lo) & (ids < self.hi)
        draw = torch.rand_like(ids, dtype=torch.float)
        mask = (draw < self.mask_ratio) & in_role & elig
        masked_input = ids.clone()
        masked_input[mask] = self.mask_id
        return masked_input, ids, mask
