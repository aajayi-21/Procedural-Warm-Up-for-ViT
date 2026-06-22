"""Train/eval loops for downstream image classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from timm.utils import accuracy


def _to_device(samples, targets, device, upsample_to):
    """Move a batch to ``device`` and (optionally) upscale it there.

    Upscaling on the GPU keeps the host->device transfer tiny (e.g. 32px instead of 224px,
    ~50x less data and pin-memory copy), which is what unblocks the starved GPU.
    """
    samples = samples.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    if upsample_to is not None and samples.shape[-1] != upsample_to:
        samples = F.interpolate(samples, size=upsample_to, mode="bicubic",
                                align_corners=False)
    return samples, targets


def train_one_epoch(model, loader, optimizer, criterion, device, *,
                    mixup_fn=None, scaler=None, use_amp=False, clip_grad=1.0,
                    upsample_to=None, progress=False, desc="train") -> float:
    model.train()
    device_type = device.type
    # Accumulate on-GPU and sync once per epoch — avoids a host<->device stall every step,
    # which otherwise serializes compute and data transfer (the dominant cost for a tiny
    # model). The CPU can race ahead queuing batches so the GPU stays fed.
    loss_sum = torch.zeros((), device=device)
    n_total = 0
    pbar = None
    if progress:
        from tqdm.auto import tqdm
        pbar = tqdm(total=len(loader), dynamic_ncols=True, desc=desc, leave=False)
    for step, (samples, targets) in enumerate(loader):
        samples, targets = _to_device(samples, targets, device, upsample_to)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            output = model(samples)
            loss = criterion(output, targets)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        bs = samples.size(0)
        loss_sum += loss.detach() * bs
        n_total += bs
        if pbar is not None:
            pbar.update(1)
            if (step + 1) % 50 == 0:  # cheap periodic sync for the live loss readout
                pbar.set_postfix(loss=f"{(loss_sum / max(n_total, 1)).item():.3f}")
    if pbar is not None:
        pbar.close()
    return (loss_sum / max(n_total, 1)).item()


@torch.no_grad()
def evaluate(model, loader, device, *, use_amp=False, upsample_to=None) -> dict:
    model.eval()
    device_type = device.type
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    loss_sum = torch.zeros((), device=device)
    top1_sum = torch.zeros((), device=device)
    top5_sum = torch.zeros((), device=device)
    n_total = 0
    for samples, targets in loader:
        samples, targets = _to_device(samples, targets, device, upsample_to)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            output = model(samples)
            loss_sum += criterion(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))  # per-batch percentages
        bs = samples.size(0)
        top1_sum += acc1 * bs
        top5_sum += acc5 * bs
        n_total += bs
    n = max(n_total, 1)
    return {"top1": (top1_sum / n).item(), "top5": (top5_sum / n).item(),
            "loss": (loss_sum / n).item()}
