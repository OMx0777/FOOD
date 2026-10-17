"""Render synthetic back-of-pack labels with KNOWN values, for testing the OCR pipeline.

These are clearly synthetic (file names start with "synthetic_") and exist so the
OCR test bench works out of the box. Replace / extend them with real photos of
real packs (see README - "Custom OCR test bench").

Each label is distorted like a phone photo: slight rotation, perspective, blur,
noise, uneven lighting and a glare patch.

Usage:
    python scripts/make_synthetic_labels.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "dataset_ocr" / "raw_images"
GT_PATH = ROOT / "dataset_ocr" / "ground_truth_labels.json"

LABELS = [
    dict(name="Crunchy Masala Chips", category="packaged_solid", basis="g", serve=30,
         energy=545, fat=34.1, sat=15.2, sugar=2.4, added=1.1, sodium_mg=740, mfd="05/2025", exp="02/2026"),
    dict(name="Oat Digestive Biscuits", category="packaged_solid", basis="g", serve=None,
         energy=462, fat=18.0, sat=3.1, sugar=2.6, added=2.0, sodium_mg=410, mfd="08/2026", exp="05/2027"),
    dict(name="Mango Fruit Drink", category="packaged_liquid", basis="ml", serve=200,
         energy=58, fat=0.0, sat=0.0, sugar=14.2, added=12.8, sodium_mg=12, mfd="09/2026", exp="03/2027"),
    dict(name="Lightly Salted Makhana", category="packaged_solid", basis="g", serve=25,
         energy=398, fat=9.5, sat=1.2, sugar=0.8, added=0.0, sodium_mg=520, mfd="07/2026", exp="01/2027"),
    dict(name="Unsweetened Soy Milk", category="packaged_liquid", basis="ml", serve=250,
         energy=33, fat=1.8, sat=0.3, sugar=0.4, added=0.0, sodium_mg=45, mfd="10/2026", exp="04/2027"),
    dict(name="Classic Cream Wafers", category="packaged_solid", basis="g", serve=None, salt=True,
         energy=520, fat=27.0, sat=16.4, sugar=38.0, added=36.5, sodium_mg=200, mfd="01/2026", exp="07/2026"),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"])
    for name in names:
        for folder in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"):
            try:
                return ImageFont.truetype(f"{folder}/{name}", size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def render(spec: dict) -> Image.Image:
    w, h = 1100, 1350
    img = Image.new("RGB", (w, h), (250, 248, 240))
    d = ImageDraw.Draw(img)
    d.text((60, 40), spec["name"].upper(), font=font(58, True), fill=(150, 20, 20))
    d.text((60, 120), "NUTRITIONAL INFORMATION (Approx.)", font=font(40, True), fill=(20, 20, 20))
    unit = spec["basis"]
    header = f"Per 100 {unit}" + (f"      Per serve ({spec['serve']} {unit})" if spec["serve"] else "")
    d.text((560, 200), header, font=font(32, True), fill=(20, 20, 20))
    d.line((60, 250, w - 60, 250), fill=(20, 20, 20), width=3)

    sodium_row = (("Salt (g)", spec["sodium_mg"] / 1000 * 2.5, "g") if spec.get("salt")
                  else ("Sodium (mg)", spec["sodium_mg"], "mg"))
    rows = [("Energy (kcal)", spec["energy"], "kcal"), ("Total Fat (g)", spec["fat"], "g"),
            ("Saturated Fat (g)", spec["sat"], "g"), ("Total Sugars (g)", spec["sugar"], "g"),
            ("Added Sugars (g)", spec["added"], "g"), sodium_row]
    y = 275
    for label, value, u in rows:
        d.text((60, y), label, font=font(36), fill=(20, 20, 20))
        fmt = (lambda v: f"{v:.0f}") if u in ("mg", "kcal") else (lambda v: f"{v:.1f}")
        d.text((600, y), fmt(value), font=font(36), fill=(20, 20, 20))
        if spec["serve"]:
            d.text((900, y), fmt(value * spec["serve"] / 100), font=font(36), fill=(20, 20, 20))
        y += 70
        d.line((60, y - 12, w - 60, y - 12), fill=(140, 140, 140), width=1)

    d.text((60, y + 30), "INGREDIENTS: Refined wheat flour, edible vegetable oil, sugar,", font=font(28), fill=(40, 40, 40))
    d.text((60, y + 70), "iodised salt, spices and condiments, emulsifier (INS 322).", font=font(28), fill=(40, 40, 40))
    d.text((60, y + 150), f"MFD: {spec['mfd']}    EXP: {spec['exp']}    B.No: K2{random.randint(100, 999)}",
           font=font(38, True), fill=(10, 10, 10))
    d.text((60, y + 210), "MRP Rs. 20.00 (Incl. of all taxes)", font=font(28), fill=(40, 40, 40))
    return img


def photograph(img: Image.Image, rng: random.Random) -> Image.Image:
    """Make a clean render look like a hand-held phone photo."""
    w, h = img.size
    dx, dy = rng.randint(10, 45), rng.randint(10, 45)
    coeffs = _perspective_coeffs([(0, 0), (w, 0), (w, h), (0, h)],
                                 [(dx, dy), (w - rng.randint(0, 30), rng.randint(0, 25)),
                                  (w - rng.randint(5, 40), h - rng.randint(5, 40)), (rng.randint(0, 30), h - dy)])
    img = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=(90, 90, 90))
    img = img.rotate(rng.uniform(-3, 3), resample=Image.BICUBIC, expand=False, fillcolor=(90, 90, 90))
    arr = np.asarray(img).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    light = 0.82 + 0.25 * (xx / w) * (1 - 0.4 * yy / h)                      # uneven lighting
    cx, cy = rng.randint(200, w - 200), rng.randint(300, h - 300)
    glare = 90 * np.exp(-(((xx - cx) / 120) ** 2 + ((yy - cy) / 60) ** 2))  # specular highlight
    arr = arr * light[..., None] + glare[..., None] + rng.gauss(0, 1) * np.random.default_rng(1).normal(0, 6, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.2)))


def _perspective_coeffs(src, dst):
    matrix = []
    for (x, y), (u, v) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(matrix, dtype=np.float64)
    b = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(a, b).tolist()


def main():
    rng = random.Random(7)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = json.loads(GT_PATH.read_text(encoding="utf-8")) if GT_PATH.exists() else {"labels": []}
    entries = [e for e in existing["labels"] if not e["image"].startswith("synthetic_")]
    for i, spec in enumerate(LABELS, 1):
        name = f"synthetic_{i:02d}_{spec['name'].lower().replace(' ', '_')}.jpg"
        photograph(render(spec), rng).save(OUT_DIR / name, quality=88)
        entries.append({
            "image": name,
            "product_name": spec["name"],
            "category": spec["category"],
            "synthetic": True,
            "per_100": {
                "added_sugar_g": spec["added"],
                "total_sugar_g": spec["sugar"],
                "saturated_fat_g": spec["sat"],
                "sodium_g": round(spec["sodium_mg"] / 1000, 4),
            },
        })
        print("wrote", name)
    out = {"_comment": existing.get("_comment", ""), "labels": entries}
    GT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Ground truth: {GT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
