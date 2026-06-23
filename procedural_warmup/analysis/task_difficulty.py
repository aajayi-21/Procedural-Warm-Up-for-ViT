"""Quantify how hard each procedural warm-up task is (no trained model needed).

A warm-up only transfers if its masked-token objective forces the network to learn real
structure. If the masked targets are low-entropy and a trivial predictor already nails
them, the task is "too easy" and the gradient is dominated by shallow patterns — exactly
the WW failure mode in the paper. This computes, per source, over freshly generated data:

- mask%            : fraction of tokens masked (the supervision density);
- distinct targets : number of distinct token values the model must choose among;
- entropy (bits)   : Shannon entropy of the masked-target distribution (higher = harder);
- unigram acc      : accuracy of always predicting the most frequent masked target
                     (a floor on difficulty — high = lots of free correct answers).

    python -m procedural_warmup.analysis.task_difficulty
"""

from __future__ import annotations

import argparse
import copy

import torch

from procedural_warmup.config import load_config
from procedural_warmup.data import build_source

_CFG = "procedural_warmup/config/files"


def task_stats(cfg, n_batches: int = 8, batch_size: int = 256) -> dict:
    dataset, masking = build_source(cfg)
    targets = []
    mask_frac = 0.0
    for _ in range(n_batches):
        batch = torch.stack([dataset[i] for i in range(batch_size)])
        _, target, mask = masking(batch)
        targets.append(target[mask])
        mask_frac += mask.float().mean().item()
    t = torch.cat(targets)
    counts = torch.bincount(t)
    probs = counts.float() / counts.sum()
    nz = probs[probs > 0]
    return {
        "mask_pct": 100 * mask_frac / n_batches,
        "distinct": int((counts > 0).sum()),
        "entropy_bits": float(-(nz * nz.log2()).sum()),
        "unigram_acc": float(probs.max()),
    }


def _variants() -> dict:
    """A panel of warm-up tasks: the grammars plus CA tokenization/masking variants."""
    dyck = load_config(f"{_CFG}/dyck-vit-t.yaml")
    dyck_shuffle = load_config(f"{_CFG}/dyck-shuffle.yaml")
    ww = load_config(f"{_CFG}/ww.yaml")
    ca = load_config(f"{_CFG}/ca-rule110.yaml")  # binary + random masking (current default)

    def tweak(base, **kw):
        c = copy.deepcopy(base)
        if "mask_mode" in kw:
            c.masking.mode = kw["mask_mode"]
        if "tok_mode" in kw:
            c.ca.tokenize.mode = kw["tok_mode"]
            c.vocab.K = 130 if kw["tok_mode"] == "block" else c.vocab.K
        return c

    return {
        "dyck (CF)": dyck,
        "dyck_shuffle (CS)": dyck_shuffle,
        "ww (regular)": ww,
        "ca binary+random": ca,
        "ca binary+forward": tweak(ca, mask_mode="forward"),
        "ca block+random": tweak(ca, tok_mode="block"),
        "ca block+forward": tweak(ca, tok_mode="block", mask_mode="forward"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Warm-up task difficulty diagnostic.")
    ap.add_argument("--batches", type=int, default=8)
    args = ap.parse_args()

    rows = {name: task_stats(cfg, n_batches=args.batches) for name, cfg in _variants().items()}
    print(f"{'task':22} {'mask%':>6} {'distinct':>9} {'entropy(bits)':>14} {'unigram-acc':>12}")
    print("-" * 68)
    for name, s in rows.items():
        print(f"{name:22} {s['mask_pct']:6.1f} {s['distinct']:9d} "
              f"{s['entropy_bits']:14.2f} {s['unigram_acc']:12.3f}")


if __name__ == "__main__":
    main()
