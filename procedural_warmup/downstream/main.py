"""Entry point for downstream image-classification training/transfer.

    python -m procedural_warmup.downstream.main --config procedural_warmup/config/files/downstream-cifar.yaml

Builds a fresh timm ViT, optionally initializes its transformer blocks from a stripped
warm-up checkpoint (``cfg.init_checkpoint``), trains with the DeiT-style recipe (AdamW +
cosine, RandAugment, Mixup/CutMix, label smoothing, AMP), and writes per-epoch logs, a
final ``metrics.json``, a training-curve figure and a markdown report.
"""

from __future__ import annotations

import argparse

import timm
import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler

from procedural_warmup.config import config_to_dict, load_downstream_config
from procedural_warmup.downstream.datasets import build_loaders
from procedural_warmup.downstream.engine import evaluate, train_one_epoch
from procedural_warmup.downstream.init_weights import load_warmup_init
from procedural_warmup.utils import RunDir, now_iso, resolve_device, set_seed


def build_criterion(cfg, mixup_fn):
    if mixup_fn is not None:
        return SoftTargetCrossEntropy()
    if cfg.aug.smoothing > 0:
        return LabelSmoothingCrossEntropy(smoothing=cfg.aug.smoothing)
    return torch.nn.CrossEntropyLoss()


def run(cfg) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.train.device)
    if device.type == "cuda":
        # Fixed input size -> let cuDNN pick the fastest kernels; enable TF32 matmuls.
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"[device] training on GPU: {torch.cuda.get_device_name(device)} "
              f"(AMP={cfg.train.use_amp})")
    else:
        print("[device] WARNING: training on CPU — the GPU is NOT in use. "
              "Check `torch.cuda.is_available()` and your CUDA/torch install.")
    run_dir = RunDir.create(cfg.results_dir, cfg.run_name)
    run_dir.save_config(config_to_dict(cfg))

    train_loader, val_loader, n_classes, mixup_fn = build_loaders(cfg)

    model = timm.create_model(
        cfg.model.name,
        pretrained=False,
        num_classes=n_classes,
        drop_path_rate=cfg.model.drop_path_rate,
    )
    init_summary = None
    if cfg.init_checkpoint:
        init_summary = load_warmup_init(model, cfg.init_checkpoint)
    model.to(device)

    optimizer = create_optimizer_v2(
        model, opt="adamw", lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=cfg.train.epochs,
        lr_min=cfg.train.min_lr,
        warmup_t=cfg.train.warmup_epochs,
        warmup_lr_init=cfg.train.warmup_lr,
    )
    criterion = build_criterion(cfg, mixup_fn)
    use_amp = cfg.train.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    from tqdm.auto import tqdm

    from procedural_warmup.analysis.figures import plot_downstream_curve

    progress = cfg.logging.progress
    best_top1 = 0.0
    val = {"top1": 0.0, "top5": 0.0, "loss": float("nan")}
    epoch_bar = tqdm(range(cfg.train.epochs), disable=not progress, dynamic_ncols=True,
                     desc=f"train:{cfg.run_name}")
    for epoch in epoch_bar:
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            mixup_fn=mixup_fn, scaler=scaler, use_amp=use_amp,
            clip_grad=cfg.train.clip_grad,
            progress=progress, desc=f"epoch {epoch}/{cfg.train.epochs}",
        )
        do_eval = ((epoch + 1) % cfg.train.eval_interval == 0
                   or epoch == cfg.train.epochs - 1)
        if do_eval:
            val = evaluate(model, val_loader, device, use_amp=use_amp)
            best_top1 = max(best_top1, val["top1"])
            run_dir.log_row({
                "epoch": epoch, "lr": lr, "train_loss": train_loss,
                "val_top1": val["top1"], "val_top5": val["top5"], "val_loss": val["loss"],
            })
            # Refresh the live downstream curve so progress is visible mid-run.
            plot_downstream_curve(cfg.results_dir, cfg.run_name, run_dir.figures_dir)
            msg = (f"epoch {epoch:03d} | lr {lr:.2e} | train_loss {train_loss:.4f} | "
                   f"top1 {val['top1']:.2f} | top5 {val['top5']:.2f} | best {best_top1:.2f}")
            if progress:
                epoch_bar.set_postfix(top1=f"{val['top1']:.2f}", best=f"{best_top1:.2f}")
                tqdm.write(msg)
            else:
                print(msg)
        elif not progress:
            print(f"epoch {epoch:03d} | lr {lr:.2e} | train_loss {train_loss:.4f}")
    epoch_bar.close()

    metrics = {
        "run_name": cfg.run_name,
        "dataset": cfg.data.dataset,
        "init_checkpoint": cfg.init_checkpoint,
        "epochs": cfg.train.epochs,
        "best_top1": best_top1,
        "final_top1": val["top1"],
        "final_top5": val["top5"],
        "init_summary": init_summary,
        "finished": now_iso(),
    }
    run_dir.save_metrics(metrics)
    _finalize_report(cfg, run_dir, metrics)
    return metrics


def _finalize_report(cfg, run_dir: RunDir, metrics: dict) -> None:
    from procedural_warmup.analysis.figures import plot_downstream_curve

    fig = plot_downstream_curve(cfg.results_dir, cfg.run_name, run_dir.figures_dir)
    init_desc = cfg.init_checkpoint or "random init (no warm-up)"
    run_dir.write_report(
        title=f"Downstream run: {cfg.run_name}",
        sections={
            "Configuration": (
                f"- dataset: **{cfg.data.dataset}** ({cfg.data.input_size}px)\n"
                f"- init: `{init_desc}`\n"
                f"- model: {cfg.model.name}, {cfg.train.epochs} epochs, "
                f"AdamW lr={cfg.train.lr} wd={cfg.train.weight_decay}\n"
                f"- aug: mixup={cfg.aug.mixup} cutmix={cfg.aug.cutmix} "
                f"smoothing={cfg.aug.smoothing}"
            ),
            "Result": (
                f"Best top-1: **{metrics['best_top1']:.2f}%** "
                f"(final {metrics['final_top1']:.2f}%, top-5 {metrics['final_top5']:.2f}%)."
            ),
        },
        figures=[str(fig)] if fig else None,
    )
    print(f"[report] wrote {run_dir.root / 'report.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Downstream image-classification transfer.")
    ap.add_argument("--config", required=True, help="path to a downstream YAML config")
    ap.add_argument("--init", default=None, help="override cfg.init_checkpoint")
    ap.add_argument("--dataset", default=None, help="override cfg.data.dataset")
    ap.add_argument("--run-name", default=None, help="override cfg.run_name")
    args = ap.parse_args()

    cfg = load_downstream_config(args.config)
    if args.init is not None:
        cfg.init_checkpoint = None if args.init.lower() == "none" else args.init
    if args.dataset is not None:
        cfg.data.dataset = args.dataset
    if args.run_name is not None:
        cfg.run_name = args.run_name
    run(cfg)


if __name__ == "__main__":
    main()
