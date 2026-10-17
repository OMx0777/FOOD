"""Lazy, thread-safe loaders for the heavy models (CLIP, EasyOCR, MobileNetV2).

Each model is created on first use and cached for the life of the process, so
the Flask server pays the start-up cost only once.
"""
import json
import os
import threading
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as tv_models

from . import config


def _clip_cached() -> bool:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))
    return any((hub / f"models--{config.CLIP_MODEL_NAME.replace('/', '--')}" / "snapshots").glob("*/config.json"))


# Once CLIP is on disk, keep the Hugging Face libraries offline. Even with local_files_only they
# otherwise call the Hub on every start (a background safetensors-conversion check). This must be set
# before huggingface_hub is first imported, which happens lazily in get_clip().
if _clip_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_lock = threading.Lock()
_cache: dict = {}


class ModelNotTrainedError(RuntimeError):
    pass


def build_mobilenet_v2(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = tv_models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = tv_models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    return model


def get_spoilage_model():
    """Return (model, class_names). Raises ModelNotTrainedError if weights are missing."""
    with _lock:
        if "spoilage" not in _cache:
            if not config.SPOILAGE_WEIGHTS.exists() or not config.SPOILAGE_CLASSES.exists():
                raise ModelNotTrainedError(
                    "Spoilage model not trained yet. Run: python scripts/train_vision.py"
                )
            class_names = json.loads(config.SPOILAGE_CLASSES.read_text(encoding="utf-8"))
            model = build_mobilenet_v2(len(class_names), pretrained=False)
            state = torch.load(config.SPOILAGE_WEIGHTS, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval().to(DEVICE)
            _cache["spoilage"] = (model, class_names)
        return _cache["spoilage"]


def get_clip():
    """Return (model, processor) for zero-shot guardrail classification."""
    with _lock:
        if "clip" not in _cache:
            from transformers import CLIPModel, CLIPProcessor

            # Use the copy already on disk (works offline, no Hub round-trips); download only the first time.
            try:
                model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME, local_files_only=True)
                processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME, local_files_only=True)
            except OSError:
                model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
                processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
            model = model.eval().to(DEVICE)
            _cache["clip"] = (model, processor)
        return _cache["clip"]


def get_ocr_reader():
    with _lock:
        if "ocr" not in _cache:
            import easyocr

            _cache["ocr"] = easyocr.Reader(["en"], gpu=DEVICE.type == "cuda", verbose=False)
        return _cache["ocr"]


def status() -> dict:
    return {
        "device": str(DEVICE),
        "spoilage_model_trained": config.SPOILAGE_WEIGHTS.exists(),
        "loaded": sorted(_cache.keys()),
    }
