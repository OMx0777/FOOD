"""Fine-tune MobileNetV2 (ImageNet weights) on the fresh/stale produce dataset.

Two-phase transfer learning:
  1. Freeze the convolutional backbone, train only the new classifier head.
  2. Unfreeze everything and fine-tune with a lower, cosine-decayed learning rate.

The checkpoint with the best validation macro-F1 is kept, then evaluated once on
the held-out test split (see evaluate_vision.py).

Usage:
    python scripts/train_vision.py [--epochs-head 3] [--epochs-finetune 12] [--batch-size 64]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foodguard import config  # noqa: E402
from foodguard.models import DEVICE, build_mobilenet_v2  # noqa: E402
from foodguard.vision import EVAL_TRANSFORM, TRAIN_TRANSFORM  # noqa: E402


def rgb_loader(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def make_loader(split: str, transform, batch_size: int, shuffle: bool):
    ds = datasets.ImageFolder(config.DATASET_VISION_DIR / split, transform=transform, loader=rgb_loader)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=6,
                    pin_memory=True, persistent_workers=True)
    return ds, dl


def run_epoch(model, loader, criterion, optimizer=None, scheduler=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(DEVICE.type, dtype=torch.float16,
                                                              enabled=DEVICE.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        total_loss += loss.item() * images.size(0)
        preds.append(logits.argmax(1).cpu())
        targets.append(labels.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": total_loss / len(targets),
        "accuracy": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-head", type=int, default=3)
    parser.add_argument("--epochs-finetune", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config.MODELS_DIR.mkdir(exist_ok=True)

    train_ds, train_dl = make_loader("train", TRAIN_TRANSFORM, args.batch_size, shuffle=True)
    _, val_dl = make_loader("validation", EVAL_TRANSFORM, args.batch_size, shuffle=False)
    class_names = train_ds.classes
    print(f"Device: {DEVICE} | classes: {class_names} | train images: {len(train_ds)}")

    # Inverse-frequency class weights (bitter gourd has ~4x fewer images than apple).
    counts = np.bincount(train_ds.targets, minlength=len(class_names))
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    model = build_mobilenet_v2(len(class_names), pretrained=True).to(DEVICE)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    history, best_f1 = [], -1.0

    phases = [("head", args.epochs_head, args.lr_head), ("finetune", args.epochs_finetune, args.lr_finetune)]
    for phase, epochs, lr in phases:
        for p in model.features.parameters():
            p.requires_grad = phase == "finetune"
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=epochs * len(train_dl), pct_start=0.15)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_m = run_epoch(model, train_dl, criterion, optimizer, scheduler, scaler)
            val_m = run_epoch(model, val_dl, criterion)
            row = {"phase": phase, "epoch": epoch, "train": train_m, "val": val_m,
                   "seconds": round(time.time() - t0, 1)}
            history.append(row)
            flag = ""
            if val_m["macro_f1"] > best_f1:
                best_f1 = val_m["macro_f1"]
                torch.save(model.state_dict(), config.SPOILAGE_WEIGHTS)
                flag = "  <- best, saved"
            print(f"[{phase} {epoch}/{epochs}] train acc {train_m['accuracy']:.4f} loss {train_m['loss']:.4f}"
                  f" | val acc {val_m['accuracy']:.4f} F1 {val_m['macro_f1']:.4f}"
                  f" | {row['seconds']}s{flag}", flush=True)

    config.SPOILAGE_CLASSES.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (config.MODELS_DIR / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nBest validation macro-F1: {best_f1:.4f}. Weights: {config.SPOILAGE_WEIGHTS}")

    from evaluate_vision import evaluate
    evaluate("test")


if __name__ == "__main__":
    main()
