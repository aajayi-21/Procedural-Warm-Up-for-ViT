"""Load a stripped warm-up checkpoint into a fresh image-training ViT.

Only the transformer blocks and final norm are present in a stripped checkpoint; the patch
embedding, positional embedding, CLS token and classifier head stay randomly initialized.
Loading is non-strict and the missing/unexpected keys are reported so transfer is auditable.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")


def _load_state(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def _parse_blocks(blocks, depth: int) -> set:
    """``"0-3"`` | ``"all"`` | ``[0,1,2]`` -> a set of block indices."""
    if blocks in (None, "all"):
        return set(range(depth))
    if isinstance(blocks, (list, tuple, set)):
        return set(int(b) for b in blocks)
    lo, hi = str(blocks).split("-")
    return set(range(int(lo), int(hi) + 1))


def _component_match(key: str, components: str) -> bool:
    """Within a block, select the attention sub-stack, the MLP sub-stack, or all of it.

    A pre-norm ViT block is ``norm1 -> attn`` then ``norm2 -> mlp``; we group each norm with
    the sub-layer it feeds so an ``attn``/``mlp`` graft keeps the matching normalization.
    """
    if components == "all":
        return True
    if components == "attn":
        return (".attn." in key) or (".norm1." in key)
    if components == "mlp":
        return (".mlp." in key) or (".norm2." in key)
    raise ValueError(f"components must be all|attn|mlp, got {components!r}")


def _shuffle(t: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Permute a tensor's elements: preserves its value distribution / norm, destroys the
    learned structure (the necessity-test control — a shuffled block is "present but broken")."""
    flat = t.flatten()
    return flat[torch.randperm(flat.numel(), generator=gen)].view_as(t)


def apply_init_spec(model: torch.nn.Module, specs: list[dict], seed: int = 0) -> dict:
    """Surgically initialize transformer blocks from one or more warm-up checkpoints.

    ``specs`` is an ordered list of dicts (later entries override earlier on shared keys)::

        {"ckpt": <path>, "blocks": "0-3"|"all"|[..], "components": "all"|"attn"|"mlp",
         "shuffle": false, "norm": false}

    Blocks not covered by any spec keep the model's random init. The final ``norm.*`` is taken
    from the spec with ``norm=True``, else auto-assigned to the spec covering the last block.
    Enables sufficiency (init a subset), necessity (omit/shuffle a subset), component (attn vs
    MLP) and cross-checkpoint **grafts** (e.g. CA early + k-Dyck late) — all without retraining.
    """
    depth = len(model.blocks)
    gen = torch.Generator().manual_seed(seed)
    merged: dict = {}
    sources: dict = {}  # block idx -> short source tag (for the audit summary)

    explicit_norm = None
    norm_owner_block = -1
    for spec in specs:
        state = _load_state(spec["ckpt"])
        blocks = _parse_blocks(spec.get("blocks", "all"), depth)
        comp = spec.get("components", "all")
        shuffle = bool(spec.get("shuffle", False))
        tag = spec.get("tag") or Path(spec["ckpt"]).parent.name  # path-separator safe
        for key, val in state.items():
            m = _BLOCK_RE.match(key)
            if m is None:
                continue
            idx = int(m.group(1))
            if idx not in blocks or not _component_match(key, comp):
                continue
            merged[key] = _shuffle(val, gen) if shuffle else val
            sources[idx] = (tag + ("/" + comp if comp != "all" else "")
                            + ("(shuf)" if shuffle else ""))
            if idx > norm_owner_block:
                norm_owner_block = idx
                explicit_norm = (state, shuffle)
        if spec.get("norm"):
            for key, val in state.items():
                if key.startswith("norm."):
                    merged[key] = _shuffle(val, gen) if shuffle else val
            explicit_norm = None  # handled
    # auto-assign final norm to the checkpoint owning the last block
    if "norm.weight" not in merged and explicit_norm is not None:
        state, shuffle = explicit_norm
        for key, val in state.items():
            if key.startswith("norm."):
                merged[key] = _shuffle(val, gen) if shuffle else val

    incompatible = model.load_state_dict(merged, strict=False)
    summary = {
        "specs": specs,
        "loaded_tensors": len(merged),
        "block_sources": {i: sources.get(i, "random") for i in range(depth)},
        "missing": len(incompatible.missing_keys),
        "unexpected": len(incompatible.unexpected_keys),
    }
    print(f"[init-spec] {len(merged)} tensors; block sources:")
    print("           " + " ".join(f"{i}:{sources.get(i,'rand')}" for i in range(depth)))
    return summary


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
