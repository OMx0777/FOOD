"""Pipeline A: raw-produce spoilage detection with a fine-tuned MobileNetV2.

The CNN predicts one of 12 classes (fresh_<type> / stale_<type>). The decision
uses the total probability mass on "fresh" vs "stale" classes:

    P(fresh) >= 0.85  -> SAFE    (Grade A, consumable)
    P(stale) >= 0.85  -> UNSAFE  (visible rot / mould - discard)
    otherwise         -> CAUTION (early decay - cook / process immediately)

Visual proof is a Grad-CAM heatmap of the regions that drove the prediction,
plus an HSV colour measurement of dark / brown surface patches.
"""
import base64
import io
import threading

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from . import config
from .models import DEVICE, get_spoilage_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    # Hue is kept small: colour change IS the spoilage signal we want to learn.
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_gradcam_lock = threading.Lock()


def _split_class(name: str) -> tuple[str, str]:
    state, _, produce = name.partition("_")
    return state, produce


def predict_spoilage(image: Image.Image) -> dict:
    model, class_names = get_spoilage_model()
    rules = config.load_thresholds()["produce"]

    tensor = EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with _gradcam_lock:
        probs, heatmap = _forward_with_gradcam(model, tensor)

    by_type: dict[str, float] = {}
    p_fresh = p_stale = 0.0
    for name, p in zip(class_names, probs):
        state, produce = _split_class(name)
        by_type[produce] = by_type.get(produce, 0.0) + p
        if state == "fresh":
            p_fresh += p
        else:
            p_stale += p

    produce_type = max(by_type, key=by_type.get)
    if p_fresh >= rules["fresh_confidence_safe"]:
        status, grade, action = "SAFE", "A", "Good to eat as it is."
    elif p_stale >= rules["stale_confidence_unsafe"]:
        status, grade, action = "UNSAFE", "D", "Throw it away: it shows clear signs of rot."
    else:
        status, grade, action = "CAUTION", "B", "It is starting to turn, so cook or eat it today."

    top_idx = int(np.argmax(probs))
    return {
        "status": status,
        "grade": grade,
        "action": action,
        "produce_type": produce_type,
        "produce_type_probability": round(by_type[produce_type], 4),
        "fresh_probability": round(p_fresh, 4),
        "stale_probability": round(p_stale, 4),
        "top_class": class_names[top_idx],
        "class_probabilities": {n: round(p, 4) for n, p in zip(class_names, probs)},
        "surface_analysis": surface_discoloration(image),
        "heatmap_png": _overlay_heatmap(image, heatmap),
    }


def _forward_with_gradcam(model, tensor):
    """One forward/backward pass: softmax probabilities + Grad-CAM for the top class."""
    target_layer = model.features[-1]
    store = {}

    def fwd_hook(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = target_layer.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            logits = model(tensor)
            probs = logits.softmax(dim=-1)[0]
            top = int(probs.argmax())
            model.zero_grad(set_to_none=True)
            logits[0, top].backward()
    finally:
        handle.remove()

    act, grad = store["act"][0], store["grad"][0]            # (C, H, W)
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)
    return probs.detach().cpu().tolist(), cam.detach().cpu().numpy()


def _overlay_heatmap(image: Image.Image, cam: np.ndarray) -> str:
    rgb = np.array(image.convert("RGB").resize((448, 448)))
    cam_resized = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    colour = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    blended = np.uint8(0.55 * rgb + 0.45 * colour)
    return to_png_base64(blended)


def surface_discoloration(image: Image.Image) -> dict:
    """Share of the produce surface covered by dark / brown patches (HSV rule).

    Foreground = pixels whose saturation is above Otsu's threshold (produce is
    usually more saturated than the background). This is supporting evidence
    shown to the user, not part of the CNN decision.
    """
    rgb = np.array(image.convert("RGB").resize((320, 320)))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    _, fg = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    fg |= (v < 60) & (cv2.blur(fg.astype(np.uint8), (25, 25)) > 0)   # dark rot inside the fruit
    fg_pixels = int(fg.sum())
    if fg_pixels < 0.05 * fg.size:
        return {"foreground_found": False, "dark_patch_percent": None}

    dark = v < 60
    brown = (h >= 5) & (h <= 25) & (s > 70) & (v < 130)
    patches = (dark | brown) & fg
    pct = 100.0 * patches.sum() / fg_pixels
    return {"foreground_found": True, "dark_patch_percent": round(float(pct), 1)}


def to_png_base64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
