"""Layer-surgery init (Phase-2): block-range / component / graft / shuffle."""

import timm
import torch

from procedural_warmup.downstream.init_weights import (
    _component_match,
    _parse_blocks,
    apply_init_spec,
)


def test_parse_blocks():
    assert _parse_blocks("0-3", 12) == {0, 1, 2, 3}
    assert _parse_blocks("8-11", 12) == {8, 9, 10, 11}
    assert _parse_blocks("all", 12) == set(range(12))
    assert _parse_blocks([4, 5, 6, 7], 12) == {4, 5, 6, 7}


def test_component_match():
    assert _component_match("blocks.5.attn.qkv.weight", "attn")
    assert _component_match("blocks.5.norm1.weight", "attn")
    assert not _component_match("blocks.5.mlp.fc1.weight", "attn")
    assert _component_match("blocks.5.mlp.fc1.weight", "mlp")
    assert _component_match("blocks.5.norm2.weight", "mlp")
    assert _component_match("blocks.5.attn.qkv.weight", "all")


def _fake_ckpt(tmp_path, name, fill):
    m = timm.create_model("vit_tiny_patch16_224", num_classes=0)
    state = {
        k: torch.full_like(v, float(fill))
        for k, v in m.state_dict().items()
        if k.startswith("blocks.") or k.startswith("norm.")
    }
    p = tmp_path / f"{name}.pt"
    torch.save({"model": state}, p)
    return str(p)


def test_graft_block_ranges(tmp_path):
    model = timm.create_model("vit_tiny_patch16_224", num_classes=10)
    a = _fake_ckpt(tmp_path, "A", 1.0)
    b = _fake_ckpt(tmp_path, "B", 2.0)
    summ = apply_init_spec(
        model,
        [{"ckpt": a, "blocks": "0-3", "tag": "A"}, {"ckpt": b, "blocks": "8-11", "tag": "B"}],
    )
    sd = model.state_dict()
    assert torch.all(sd["blocks.0.attn.qkv.weight"] == 1.0)   # early from A
    assert torch.all(sd["blocks.11.mlp.fc1.weight"] == 2.0)   # late from B
    assert not torch.all(sd["blocks.5.attn.qkv.weight"] == 1.0)  # mid stays random
    assert not torch.all(sd["blocks.5.attn.qkv.weight"] == 2.0)
    assert torch.all(sd["norm.weight"] == 2.0)                # final norm auto from B (owns block 11)
    assert summ["block_sources"][0] == "A" and summ["block_sources"][11] == "B"
    assert summ["block_sources"][5] == "random"


def test_component_graft_attn_only(tmp_path):
    model = timm.create_model("vit_tiny_patch16_224", num_classes=10)
    a = _fake_ckpt(tmp_path, "A", 1.0)
    apply_init_spec(model, [{"ckpt": a, "blocks": "0-3", "components": "attn"}])
    sd = model.state_dict()
    assert torch.all(sd["blocks.0.attn.qkv.weight"] == 1.0)       # attn loaded
    assert not torch.all(sd["blocks.0.mlp.fc1.weight"] == 1.0)    # mlp left random


def test_shuffle_preserves_distribution(tmp_path):
    model = timm.create_model("vit_tiny_patch16_224", num_classes=10)
    m2 = timm.create_model("vit_tiny_patch16_224", num_classes=0)
    state = {k: v.clone() for k, v in m2.state_dict().items() if k.startswith("blocks.0.attn.qkv")}
    p = tmp_path / "R.pt"
    torch.save({"model": state}, p)
    apply_init_spec(model, [{"ckpt": str(p), "blocks": "0-0", "shuffle": True}])
    loaded = model.state_dict()["blocks.0.attn.qkv.weight"]
    orig = state["blocks.0.attn.qkv.weight"]
    assert torch.allclose(loaded.flatten().sort().values, orig.flatten().sort().values)  # same values
    assert not torch.equal(loaded, orig)  # different arrangement
