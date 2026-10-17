"""Evaluate the trained spoilage model on a held-out split.

Writes to models/:
  * vision_metrics_<split>.json  - accuracy, per-class report, fresh-vs-stale accuracy,
                                    and how the SAFE / CAUTION / UNSAFE rule behaves
  * confusion_matrix_<split>.png - 12-class confusion matrix

Usage:
    python scripts/evaluate_vision.py [--split test]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from foodguard import config  # noqa: E402
from foodguard.models import DEVICE, get_spoilage_model  # noqa: E402
from foodguard.vision import EVAL_TRANSFORM  # noqa: E402
from train_vision import rgb_loader  # noqa: E402


def evaluate(split: str = "test") -> dict:
    model, class_names = get_spoilage_model()
    ds = datasets.ImageFolder(config.DATASET_VISION_DIR / split, transform=EVAL_TRANSFORM, loader=rgb_loader)
    assert ds.classes == class_names, "Class order mismatch between dataset and checkpoint"
    dl = DataLoader(ds, batch_size=128, num_workers=4)

    probs, targets = [], []
    with torch.no_grad():
        for images, labels in dl:
            probs.append(model(images.to(DEVICE)).softmax(1).cpu())
            targets.append(labels)
    probs, targets = torch.cat(probs).numpy(), torch.cat(targets).numpy()
    preds = probs.argmax(1)

    fresh_idx = [i for i, n in enumerate(class_names) if n.startswith("fresh_")]
    p_fresh = probs[:, fresh_idx].sum(1)
    true_fresh = np.isin(targets, fresh_idx)

    rules = config.load_thresholds()["produce"]
    decision = np.where(p_fresh >= rules["fresh_confidence_safe"], "SAFE",
                        np.where(1 - p_fresh >= rules["stale_confidence_unsafe"], "UNSAFE", "CAUTION"))
    decided = decision != "CAUTION"
    decision_correct = (decision == "SAFE") == true_fresh

    metrics = {
        "split": split,
        "images": int(len(targets)),
        "accuracy_12_class": float((preds == targets).mean()),
        "accuracy_fresh_vs_stale": float(((p_fresh >= 0.5) == true_fresh).mean()),
        "decision_rule": {
            "safe": int((decision == "SAFE").sum()),
            "caution": int((decision == "CAUTION").sum()),
            "unsafe": int((decision == "UNSAFE").sum()),
            "accuracy_when_decided": float(decision_correct[decided].mean()) if decided.any() else None,
            "stale_marked_safe": int(((decision == "SAFE") & ~true_fresh).sum()),
            "fresh_marked_unsafe": int(((decision == "UNSAFE") & true_fresh).sum()),
        },
        "per_class": classification_report(targets, preds, target_names=class_names,
                                           output_dict=True, zero_division=0),
    }
    out = config.MODELS_DIR / f"vision_metrics_{split}.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=60, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"MobileNetV2 spoilage - {split} split (acc {metrics['accuracy_12_class']:.3f})")
    fig.tight_layout()
    fig.savefig(config.MODELS_DIR / f"confusion_matrix_{split}.png", dpi=130)

    print(f"\n== {split} ({metrics['images']} images) ==")
    print(f"12-class accuracy      : {metrics['accuracy_12_class']:.4f}")
    print(f"Fresh-vs-stale accuracy: {metrics['accuracy_fresh_vs_stale']:.4f}")
    print(f"Decision rule          : {metrics['decision_rule']}")
    print(classification_report(targets, preds, target_names=class_names, digits=3, zero_division=0))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    evaluate(parser.parse_args().split)
