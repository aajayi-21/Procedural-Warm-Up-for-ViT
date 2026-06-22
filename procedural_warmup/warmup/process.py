"""Strip a warm-up checkpoint for transfer to image training.

Removes everything that is procedural-warm-up-specific (frozen token/positional embeddings,
the MLM head) and everything re-initialized for vision (patch embedding, the backbone's own
positional embedding, CLS token, classifier head). What remains — the transformer blocks
and the final norm — are the weights that carry the learned structure into image training.

Usage::

    python -m procedural_warmup.warmup.process checkpoints/ca-rule110/ckpt_step_015000.pt
    # -> writes checkpoints/ca-rule110/ckpt_step_015000_stripped.pt  with {"model": ...}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

_SKIP_PREFIXES = ("patch_embed.", "pos_embed", "head.", "tok.", "pos.")
_SKIP_EXACT = {"cls_token", "pos_embed"}


def _extract_state_dict(payload) -> dict:
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def clean_checkpoint(src: str | Path, dst: str | Path | None = None) -> Path:
    """Strip ``src`` and write ``{"model": <blocks+norm>}`` to ``dst``."""
    src = Path(src)
    if dst is None:
        dst = src.with_name(src.stem + "_stripped.pt")
    dst = Path(dst)

    payload = torch.load(src, map_location="cpu", weights_only=False)
    raw = _extract_state_dict(payload)

    cleaned: dict = {}
    for key, val in raw.items():
        norm_key = key[4:] if key.startswith("vit.") else key
        if norm_key in _SKIP_EXACT or norm_key.startswith(_SKIP_PREFIXES):
            continue
        cleaned[norm_key] = val

    torch.save({"model": cleaned}, dst)
    print(f"[process] {len(cleaned)} tensors kept -> {dst}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description="Strip a warm-up checkpoint for transfer.")
    ap.add_argument("src", help="path to the raw warm-up checkpoint (.pt)")
    ap.add_argument("--dst", default=None, help="output path (default: <src>_stripped.pt)")
    args = ap.parse_args()
    clean_checkpoint(args.src, args.dst)


if __name__ == "__main__":
    main()
