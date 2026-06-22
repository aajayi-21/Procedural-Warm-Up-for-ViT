"""Step-based masked-token warm-up trainer.

The trainer is deliberately agnostic to the data source: it consumes a dataset and a
masking strategy (the ``(masked_input, targets, mask)`` triple) and never branches on which
procedural source produced them. It owns a :class:`RunDir`, streaming a CSV log and, on
completion, writing ``metrics.json``, a markdown report, and a training-curve figure — so
every warm-up run is self-documenting per the project's figures+report requirement.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from procedural_warmup.utils import AverageMeter, RunDir, resolve_device
from procedural_warmup.warmup.optim import make_optimizer, make_scheduler


class CheckpointManager:
    def __init__(self, cfg) -> None:
        self.save_steps = set(cfg.checkpoint.save_steps)
        self.out_dir = Path(cfg.checkpoint.out_dir) / cfg.run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def maybe_save(self, step: int, payload: dict) -> Path | None:
        if step in self.save_steps:
            path = self.out_dir / f"ckpt_step_{step:06d}.pt"
            torch.save(payload, path)
            print(f"[ckpt] saved {path}")
            return path
        return None


class Trainer:
    def __init__(self, cfg, model, mlm_head, dataset, masking, run_dir: RunDir,
                 logger=None) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.training.device)
        self.model = model.to(self.device)
        self.mlm_head = mlm_head.to(self.device)
        self.masking = masking
        self.run_dir = run_dir
        self.logger = logger

        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")  # TF32 matmuls for attention/MLP
        loader_kwargs = {
            "num_workers": cfg.dataset.num_workers,
            "pin_memory": cfg.dataset.pin_memory,
        }
        if cfg.dataset.num_workers > 0:
            loader_kwargs["persistent_workers"] = True  # CA generation runs on the workers
        self.loader = DataLoader(
            dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
        params = [
            {"params": [p for p in self.model.parameters() if p.requires_grad]},
            {"params": self.mlm_head.parameters()},
        ]
        self.opt = make_optimizer(cfg, params)
        self.sched = make_scheduler(cfg, self.opt)
        self.ckpts = CheckpointManager(cfg)

    def _step(self, batch: torch.Tensor):
        batch = batch.to(self.device, non_blocking=True)
        masked_input, target, mask = self.masking(batch)
        feats = self.model.forward_tokens(masked_input)  # (B, N, d)
        logits = self.mlm_head(feats)  # (B, N, K)
        sel_logits = logits[mask]
        sel_targets = target[mask]
        if sel_targets.numel() == 0:
            return None
        loss = F.cross_entropy(sel_logits, sel_targets)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        if self.sched is not None:
            self.sched.step()

        acc = (sel_logits.argmax(-1) == sel_targets).float().mean().item()
        lr = self.opt.param_groups[0]["lr"]
        return loss.item(), acc, lr

    def train(self) -> Path | None:
        self.model.train()
        self.mlm_head.train()
        loss_m, acc_m = AverageMeter(), AverageMeter()
        last_ckpt: Path | None = None
        it = iter(self.loader)
        for step in range(1, self.cfg.training.steps + 1):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.loader)
                batch = next(it)
            result = self._step(batch)
            if result is None:
                continue
            loss, acc, lr = result
            loss_m.update(loss)
            acc_m.update(acc)

            if step % self.cfg.logging.print_freq == 0:
                self.run_dir.log_row(
                    {"step": step, "loss": loss, "acc": acc, "lr": lr}
                )
                print(
                    f"step {step:06d} | loss {loss:.4f} | acc {acc:.3f} | lr {lr:.2e}"
                )
                if self.logger is not None:
                    self.logger.log(step, loss, acc, lr)

            ckpt = self.ckpts.maybe_save(
                step,
                {
                    "step": step,
                    "model_state": self.model.state_dict(),
                    "mlm_head_state": self.mlm_head.state_dict(),
                    "optimizer_state": self.opt.state_dict(),
                },
            )
            last_ckpt = ckpt or last_ckpt

        self._finalize(loss_m.avg, acc_m.avg, last_ckpt)
        return last_ckpt

    def _finalize(self, avg_loss: float, avg_acc: float, ckpt: Path | None) -> None:
        """Write metrics, a training-curve figure and a markdown report."""
        metrics = {
            "run_name": self.cfg.run_name,
            "source": self.cfg.data.source,
            "rule": self.cfg.ca.rule if self.cfg.data.source in ("ca", "gol") else None,
            "masking": {"mode": self.cfg.masking.mode, "ratio": self.cfg.masking.mask_ratio},
            "steps": self.cfg.training.steps,
            "final_avg_loss": avg_loss,
            "final_avg_acc": avg_acc,
            "checkpoint": str(ckpt) if ckpt else None,
        }
        self.run_dir.save_metrics(metrics)

        # Figure: training curves regenerated from the CSV log.
        from procedural_warmup.analysis.figures import plot_warmup_curves

        fig_path = plot_warmup_curves(
            self.cfg.results_dir, self.cfg.run_name, self.run_dir.figures_dir
        )
        self.run_dir.write_report(
            title=f"Warm-up run: {self.cfg.run_name}",
            sections={
                "Configuration": (
                    f"- source: `{self.cfg.data.source}`"
                    + (f" (rule {self.cfg.ca.rule})" if self.cfg.data.source in ("ca", "gol") else "")
                    + f"\n- masking: `{self.cfg.masking.mode}` @ ratio {self.cfg.masking.mask_ratio}"
                    f"\n- steps: {self.cfg.training.steps}, batch {self.cfg.dataset.batch_size}"
                    f"\n- optimizer: AdamW lr={self.cfg.optimizer.lr} wd={self.cfg.optimizer.weight_decay}"
                ),
                "Result": (
                    f"Final masked-token loss (running avg): **{avg_loss:.4f}**; "
                    f"accuracy: **{avg_acc:.3f}**.\n\n"
                    f"Stripped checkpoint for transfer: run `process.py` on `{ckpt}`."
                ),
            },
            figures=[str(fig_path)] if fig_path else None,
        )
        print(f"[report] wrote {self.run_dir.root / 'report.md'}")
