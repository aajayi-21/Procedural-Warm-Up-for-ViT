"""Optional Weights & Biases logging for the warm-up stage (no-op unless enabled)."""

from __future__ import annotations


class WandbLogger:
    def __init__(self, cfg) -> None:
        import wandb

        self.wandb = wandb
        from procedural_warmup.config import config_to_dict

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.run_name,
            config=config_to_dict(cfg),
        )

    def log(self, step: int, loss: float, acc: float, lr: float) -> None:
        self.wandb.log({"loss": loss, "acc": acc, "lr": lr}, step=step)

    def finish(self) -> None:
        self.wandb.finish()
