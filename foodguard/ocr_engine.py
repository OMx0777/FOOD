"""Pipeline B, step 1: EasyOCR text extraction with OpenCV pre-processing.

Pre-processing (applied to phone photos of crinkled / curved / glossy packs):
  1. Rescale so small print is large enough for the text detector.
  2. Glare reduction - specular highlights (very bright, unsaturated pixels)
     are masked and in-painted from their surroundings.
  3. CLAHE contrast equalisation on the grey image.
  4. Edge-preserving bilateral denoise.
  5. A second, binarised variant (adaptive Gaussian threshold) is produced as a
     fallback for low-contrast prints.

Measured on real Open Food Facts label photos, EasyOCR keeps decimal points best
on the *unfiltered colour* image, so that is read first; the enhanced, binarised
and up-scaled variants are fallbacks whose results are merged (see pipeline.py).

EasyOCR returns word/phrase boxes; we regroup them into table rows by vertical
overlap so "Sodium ... 520 mg" stays on one line even if the words are far apart.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .models import get_ocr_reader
from .vision import to_png_base64

MIN_LONG_SIDE = 1400
MAX_LONG_SIDE = 2400


@dataclass
class OcrBox:
    text: str
    confidence: float
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class OcrLine:
    text: str
    boxes: list


def _rescale(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    long_side = max(h, w)
    if long_side < MIN_LONG_SIDE:
        scale = MIN_LONG_SIDE / long_side
        return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if long_side > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / long_side
        return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return bgr


def preprocess(image: Image.Image) -> dict[str, np.ndarray]:
    bgr = _rescale(cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    glare = ((hsv[..., 2] > 240) & (hsv[..., 1] < 40)).astype(np.uint8) * 255
    glare = cv2.dilate(glare, np.ones((5, 5), np.uint8))
    glare_ratio = float((glare > 0).mean())
    if 0 < glare_ratio < 0.25:          # large white areas are background, not glare
        gray = cv2.inpaint(gray, glare, 5, cv2.INPAINT_TELEA)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = cv2.bilateralFilter(clahe.apply(gray), 5, 40, 40)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 12)
    return {"color": bgr, "gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), "enhanced": enhanced,
            "binary": binary, "glare_ratio": glare_ratio}


def read_boxes(array: np.ndarray, **options) -> list[OcrBox]:
    reader = get_ocr_reader()
    results = reader.readtext(array, detail=1, paragraph=False, **options)
    boxes = []
    for points, text, conf in results:
        xs = [int(p[0]) for p in points]
        ys = [int(p[1]) for p in points]
        text = text.strip()
        if text:
            boxes.append(OcrBox(text, float(conf), min(xs), min(ys), max(xs), max(ys)))
    return boxes


def group_lines(boxes: list[OcrBox]) -> list[OcrLine]:
    """Cluster boxes into rows: a box joins a row when its centre lies inside the row's band."""
    rows: list[list[OcrBox]] = []
    for box in sorted(boxes, key=lambda b: b.cy):
        for row in rows:
            ref = np.median([b.cy for b in row])
            band = 0.55 * np.median([b.height for b in row])
            if abs(box.cy - ref) <= band:
                row.append(box)
                break
        else:
            rows.append([box])
    lines = []
    for row in rows:
        row.sort(key=lambda b: b.x0)
        lines.append(OcrLine(" ".join(b.text for b in row), row))
    lines.sort(key=lambda ln: min(b.y0 for b in ln.boxes))
    return lines


def prominent_text(boxes: list[OcrBox], n: int = 6) -> list[str]:
    """Largest-font words on the pack: usually the brand and product name."""
    candidates = [b for b in boxes if b.confidence > 0.3 and sum(c.isalpha() for c in b.text) >= 3]
    candidates.sort(key=lambda b: b.height, reverse=True)
    return [b.text for b in candidates[:n]]


def annotate(color_bgr: np.ndarray, boxes: list[OcrBox], highlight: set[int] = frozenset()) -> str:
    """Draw OCR boxes (highlighted ones in red) and return a base64 PNG."""
    canvas = color_bgr.copy()
    thickness = max(2, canvas.shape[1] // 600)
    for i, b in enumerate(boxes):
        colour = (40, 40, 220) if i in highlight else (60, 170, 60)
        cv2.rectangle(canvas, (b.x0, b.y0), (b.x1, b.y1), colour, thickness)
    scale = 1000 / max(canvas.shape[:2])
    if scale < 1:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return to_png_base64(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
