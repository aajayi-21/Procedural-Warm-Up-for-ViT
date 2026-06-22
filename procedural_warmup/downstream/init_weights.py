"""Load a stripped warm-up checkpoint into a fresh image-training ViT.

Only the transformer blocks and final norm are present in a stripped checkpoint; the patch
embedding, positional embedding, CLS token and classifier head stay randomly initialized.
Loading is non-strict and the missing/unexpected keys are reported so transfer is auditable.
"""

from __future__ import annotations

import torch


def load_warmup_init(model: torch.nn.Module, checkpoint_path: str,
                     layer_select: dict | None = None) -> dict:
    """Load blocks+norm from ``checkpoint_path`` into ``model`` (non-strict).

    ``layer_select`` optionally restricts which blocks transfer (Stage-5 layerwise probe):
    e.g. ``{"which": "final", "k": 4}`` — see ``analysis/layerwise.select_blocks``.
    Returns a summary dict with counts of loaded/missing/unexpected keys.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    if layer_select is not None:
        from procedural_warmup.analysis.layerwise import select_blocks

        state = select_blocks(state, **layer_select)

    incompatible = model.load_state_dict(state, strict=False)
    summary = {
        "loaded": len(state),
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
    }
    print(
        f"[init] loaded {summary['loaded']} tensors from {checkpoint_path} | "
        f"{len(summary['missing'])} missing, {len(summary['unexpected'])} unexpected"
    )
    if summary["unexpected"]:
        print(f"[init] unexpected keys (not loaded): {summary['unexpected'][:6]} ...")
    return summary
