"""The data-source registry returns valid (dataset, masking) pairs for every source."""

import torch

from procedural_warmup.data import available_sources, build_source
from procedural_warmup.data.base import MaskingStrategy, ProceduralDataset


def test_sources_registered():
    sources = available_sources()
    assert "ca" in sources
    assert "dyck" in sources
    assert "gol" in sources


def test_build_ca_source(base_cfg):
    base_cfg.data.source = "ca"
    base_cfg.vocab.K = 4
    dataset, masking = build_source(base_cfg)
    assert isinstance(dataset, ProceduralDataset)
    assert isinstance(masking, MaskingStrategy)
    sample = dataset[0]
    assert sample.shape == (base_cfg.N,)
    assert sample.dtype == torch.long
    assert int(sample.min()) >= 2 and int(sample.max()) < base_cfg.vocab.K


def test_build_dyck_source(base_cfg):
    base_cfg.data.source = "dyck"
    dataset, masking = build_source(base_cfg)
    sample = dataset[0]
    assert sample.shape == (base_cfg.N,)
    # masking returns the standard triple
    batch = sample.unsqueeze(0)
    masked, targets, mask = masking(batch)
    assert masked.shape == targets.shape == mask.shape == batch.shape


def test_unknown_source_raises(base_cfg):
    base_cfg.data.source = "does-not-exist"
    try:
        build_source(base_cfg)
        assert False, "expected KeyError"
    except KeyError:
        pass
