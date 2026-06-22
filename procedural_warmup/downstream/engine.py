"""Train/eval loops for downstream image classification."""

from __future__ import annotations

import torch
from timm.utils import accuracy

from procedural_warmup.utils import AverageMeter


def train_one_epoch(model, loader, optimizer, criterion, device, *,
                    mixup_fn=None, scaler=None, use_amp=False, clip_grad=1.0) -> float:
    model.train()
    loss_m = AverageMeter()
    device_type = device.type
    for samples, targets in loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
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

        loss_m.update(loss.item(), samples.size(0))
    return loss_m.avg


@torch.no_grad()
def evaluate(model, loader, device, *, use_amp=False) -> dict:
    model.eval()
    top1_m, top5_m, loss_m = AverageMeter(), AverageMeter(), AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    device_type = device.type
    for samples, targets in loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            output = model(samples)
            loss = criterion(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        n = samples.size(0)
        loss_m.update(loss.item(), n)
        top1_m.update(acc1.item(), n)
        top5_m.update(acc5.item(), n)
    return {"top1": top1_m.avg, "top5": top5_m.avg, "loss": loss_m.avg}
