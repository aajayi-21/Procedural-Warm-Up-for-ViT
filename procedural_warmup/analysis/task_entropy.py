"""Report warm-up *task* statistics so downstream differences can be interpreted.

A downstream gap between two warm-ups is only meaningful if it is not just explained by one
task being more entropic / harder to fit. This builds the (dataset, masking) for a warm-up
config and reports, over a sample of the prediction targets:

- target bit density and per-cell **bit entropy** (binary sources)
- target **token entropy** (Shannon entropy of the predicted-token distribution, in bits)
- mean masked fraction (1.0 for full-mask transduction)

For ``ca_step`` it also reports the **conditional** entropy proxy: the achievable target
predictability given the input. For ``mode="true"`` the target is a deterministic function of
the visible input (≈0 conditional bits — fully predictable), whereas ``mode="shuffled"`` the
target is independent of the input (conditional ≈ marginal). That gap is exactly the operator
signal the experiment isolates.

    python -m procedural_warmup.analysis.task_entropy --config <warmup.yaml> [--batches 8]
"""

from __future__ import annotations

import argparse
import math

import torch

from procedural_warmup.config import load_config
from procedural_warmup.data import build_source


def _entropy_bits(counts: torch.Tensor) -> float:
    p = counts.float()
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * (p.log() / math.log(2))).sum())


def report(config_path: str, batches: int = 8) -> dict:
    cfg = load_config(config_path)
    dataset, masking = build_source(cfg)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=0
    )
    K = cfg.vocab.K
    mode = str(getattr(cfg.ca_step, "mode", "")).lower()
    tgt_tokens, masked_frac = [], []
    it = iter(loader)
    for _ in range(batches):
        batch = next(it)
        masked_input, target, mask = masking(batch)
        tgt_tokens.append(target[mask].flatten())
        masked_frac.append(mask.float().mean().item())
    tgt = torch.cat(tgt_tokens)
    counts = torch.bincount(tgt, minlength=K)
    # binary cells live in ids {2,3}; density = fraction of state-1 among binary targets.
    binary = counts[2] + counts[3]
    density = float(counts[3] / binary) if binary > 0 else float("nan")
    bit_entropy = (
        -(density * math.log2(density) + (1 - density) * math.log2(1 - density))
        if 0 < density < 1
        else 0.0
    )
    out = {
        "config": config_path,
        "source": cfg.data.source,
        "mode": mode if cfg.data.source == "ca_step" else None,
        "n_target_tokens": int(tgt.numel()),
        "target_density(state1)": round(density, 4),
        "target_bit_entropy": round(bit_entropy, 4),
        "target_token_entropy_bits": round(_entropy_bits(counts), 4),
        "mean_masked_fraction": round(sum(masked_frac) / len(masked_frac), 4),
        "note": (
            "transduction: target deterministic given input (cond≈0)"
            if cfg.data.source == "ca_step" and mode == "true"
            else "shuffled/floor: target ~ independent of input (cond≈marginal)"
            if cfg.data.source in ("ca_step", "iid_board")
            else ""
        ),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Warm-up task entropy/density report.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", type=int, default=8)
    args = ap.parse_args()
    out = report(args.config, args.batches)
    width = max(len(k) for k in out)
    for k, v in out.items():
        print(f"{k:<{width}} : {v}")


if __name__ == "__main__":
    main()
