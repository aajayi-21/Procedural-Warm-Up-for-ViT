"""End-to-end smoke test: a few warm-up steps then checkpoint stripping.

Runs on CPU with a tiny config; confirms the full warm-up path executes, a checkpoint is
written, and ``clean_checkpoint`` keeps only transformer-block/norm weights.
"""

import torch

from procedural_warmup.data import build_source
from procedural_warmup.model import build_model
from procedural_warmup.utils import RunDir
from procedural_warmup.warmup.process import clean_checkpoint
from procedural_warmup.warmup.trainer import Trainer


def test_warmup_smoke_and_strip(base_cfg, tmp_path):
    cfg = base_cfg
    cfg.data.source = "ca"
    cfg.vocab.K = 4
    cfg.training.steps = 2
    cfg.training.device = "cpu"
    cfg.dataset.batch_size = 8
    cfg.dataset.num_workers = 0
    cfg.dataset.n_samples = 64
    cfg.logging.print_freq = 1
    cfg.scheduler.warmup_steps = 1
    cfg.checkpoint.save_steps = [2]
    cfg.checkpoint.out_dir = str(tmp_path / "checkpoints")
    cfg.results_dir = str(tmp_path / "results")

    model, mlm_head = build_model(cfg)
    dataset, masking = build_source(cfg)
    run_dir = RunDir.create(cfg.results_dir, cfg.run_name)
    trainer = Trainer(cfg, model, mlm_head, dataset, masking, run_dir)
    ckpt = trainer.train()

    assert ckpt is not None and ckpt.exists()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state" in payload and "mlm_head_state" in payload

    stripped = clean_checkpoint(ckpt)
    assert stripped.exists()
    state = torch.load(stripped, map_location="cpu", weights_only=False)["model"]
    assert len(state) > 0
    for key in state:
        assert key.startswith("blocks.") or key.startswith("norm.")
        for bad in ("patch_embed", "pos_embed", "cls_token", "head.", "tok.", "pos."):
            assert bad not in key

    # The report's metrics file was written.
    assert (run_dir.root / "metrics.json").exists()
