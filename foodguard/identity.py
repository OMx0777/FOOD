"""Stage 3: product identity verification - is this photo really the product the user named?

Packaged food
  1. Barcode: decoded from the photo (OpenCV) and/or typed by the user. Two
     different codes -> MISMATCH.
  2. Reference record from the Open Food Facts API (by barcode, else by name).
  3. Brand check: brands printed on the pack are found with fuzzy matching and
     grouped into company families (fssai_config/brand_registry.json). A brand
     from a different family than the claimed product -> MISMATCH.
  4. Nutrient cross-check: OCR values vs the reference per-100 values. Large
     disagreement on 2+ nutrients -> DIVERGENT (wrong label or OCR trouble).

Raw produce
  The named fruit/vegetable (English or common Hindi name) is compared with
  what CLIP and the CNN see. "Apple" + a photo of a tomato -> MISMATCH.
"""
import json
import re
from functools import lru_cache

import cv2
import numpy as np
import requests
from rapidfuzz import fuzz

from . import config
from .label_parser import DISPLAY_NAMES, NutrientValue

STOPWORDS = {"the", "and", "with", "pack", "packet", "new", "classic", "original", "flavour", "flavor",
             "of", "in", "for", "gm", "g", "ml", "kg", "ltr", "litre", "pouch", "bottle", "jar", "box"}

PRODUCE_ALIASES = {
    "apple": ["apple", "seb", "saib"],
    "banana": ["banana", "kela"],
    "bitter_gourd": ["bitter gourd", "karela", "bitter melon"],
    "capsicum": ["capsicum", "bell pepper", "shimla mirch"],
    "orange": ["orange", "santra", "narangi", "mosambi"],
    "tomato": ["tomato", "tamatar"],
    "mango": ["mango", "aam"],
    "potato": ["potato", "aloo"],
    "onion": ["onion", "pyaz", "kanda"],
    "brinjal": ["brinjal", "baingan", "eggplant"],
    "okra": ["okra", "bhindi", "lady finger"],
    "cauliflower": ["cauliflower", "gobi"],
    "grapes": ["grapes", "angoor"],
    "carrot": ["carrot", "gajar"],
    "cucumber": ["cucumber", "kheera"],
    "lemon": ["lemon", "nimbu"],
    "guava": ["guava", "amrood"],
    "pomegranate": ["pomegranate", "anar"],
}

OFF_FIELDS = "code,product_name,brands,nutriments,image_front_url,quantity,categories_tags"


# ------------------------------------------------------------------ barcode --

def decode_barcode(bgr: np.ndarray) -> str | None:
    detector = cv2.barcode.BarcodeDetector()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    for img in (bgr, gray, cv2.resize(gray, None, fx=0.5, fy=0.5)):
        try:
            ok, infos, _types, _pts = detector.detectAndDecodeWithType(img)
        except cv2.error:
            continue
        if ok:
            codes = [c for c in infos if c and c.isdigit() and len(c) in (8, 12, 13, 14)]
            if codes:
                return codes[0]
    return None


# ----------------------------------------------------------- open food facts --

def _off_get(url: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(url, params=params, headers={"User-Agent": config.HTTP_USER_AGENT},
                         timeout=config.HTTP_TIMEOUT_S)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


_off_cache: dict = {}      # only successful look-ups are cached, so a network blip is retried


def off_by_barcode(code: str) -> dict | None:
    key = ("barcode", code)
    if key not in _off_cache:
        data = _off_get(f"{config.OPEN_FOOD_FACTS_URL}/api/v2/product/{code}.json", {"fields": OFF_FIELDS})
        if data is None:
            return None
        _off_cache[key] = data["product"] if data.get("status") == 1 else None
    return _off_cache[key]


def _brand_text(product: dict) -> str:
    brands = product.get("brands") or ""
    return ", ".join(brands) if isinstance(brands, list) else brands


def off_search(name: str) -> dict | None:
    """Best name match: search-a-licious API first, legacy search.pl as fallback."""
    key = ("search", name.lower())
    if key in _off_cache:
        return _off_cache[key]
    data = _off_get(config.OPEN_FOOD_FACTS_SEARCH_URL, {"q": name, "page_size": 10, "fields": OFF_FIELDS})
    products = (data or {}).get("hits")
    if products is None:
        data = _off_get(f"{config.OPEN_FOOD_FACTS_URL}/cgi/search.pl",
                        {"search_terms": name, "search_simple": 1, "action": "process", "json": 1,
                         "page_size": 10, "fields": OFF_FIELDS})
        products = (data or {}).get("products")
    if products is None:
        return None                                    # offline: do not cache
    min_score = config.load_thresholds()["guardrails"]["identity_fuzzy_min_score"]
    def text(p):
        return f"{_brand_text(p)} {p.get('product_name', '')}"

    # Most of the claimed name's words must appear in the record ("Amul Butter" must not match
    # "Amul Milk Chocolate" just because the brand word is shared).
    scored = sorted(((fuzz.token_set_ratio(name.lower(), text(p).lower()), i) for i, p in enumerate(products)
                     if name_visible(name, text(p)) >= 0.75), reverse=True)
    matches = [products[i] for score, i in scored if score >= min_score]
    result = None
    # Crowd-sourced records are often incomplete: among the best few name matches, prefer one whose
    # full record (product endpoint) has a nutrition table to cross-check against.
    for prod in matches[:4]:
        full = off_by_barcode(prod["code"]) if prod.get("code") else None
        candidate = {**prod, **(full or {})}
        candidate["brands"] = _brand_text(candidate)
        if result is None:
            result = candidate
        if reference_nutrients(candidate):
            result = candidate
            break
    _off_cache[key] = result
    return result


def reference_nutrients(product: dict) -> dict[str, float]:
    n = product.get("nutriments") or {}
    out = {}
    for key, off_key in (("added_sugar_g", "added-sugars_100g"), ("total_sugar_g", "sugars_100g"),
                         ("saturated_fat_g", "saturated-fat_100g"), ("total_fat_g", "fat_100g"),
                         ("sodium_g", "sodium_100g")):
        try:
            if n.get(off_key) not in (None, ""):
                out[key] = float(n[off_key])
        except (TypeError, ValueError):
            pass
    if "sodium_g" not in out and n.get("salt_100g") not in (None, ""):
        try:
            out["sodium_g"] = float(n["salt_100g"]) / 2.5
        except (TypeError, ValueError):
            pass
    return out


# ------------------------------------------------------------------- brands --

@lru_cache(maxsize=1)
def _brand_registry() -> dict[str, str]:
    data = json.loads((config.ROOT / "fssai_config" / "brand_registry.json").read_text(encoding="utf-8"))
    return {brand: family for family, brands in data["families"].items() for brand in brands}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9&'+ ]", " ", text.lower())).strip()


def find_brands(text: str) -> dict[str, str]:
    """Brands mentioned in text -> their family. Short names need an exact word match."""
    norm = f" {_norm(text)} "
    found = {}
    for brand, family in _brand_registry().items():
        b = _norm(brand)
        if re.search(rf"(?<![a-z0-9]){re.escape(b)}(?![a-z0-9])", norm):
            found[brand] = family
        elif len(b) >= 6 and fuzz.partial_ratio(b, norm) >= 92:
            found[brand] = family
    return found


def _name_tokens(name: str) -> list[str]:
    return [t for t in _norm(name).split() if len(t) >= 3 and t not in STOPWORDS and not t.isdigit()]


def name_visible(name: str, ocr_text: str) -> float:
    """Fraction of the claimed name's significant words that appear (fuzzily) in the OCR text."""
    tokens = _name_tokens(name)
    if not tokens:
        return 0.0
    text = _norm(ocr_text)
    hits = sum(1 for t in tokens if fuzz.partial_ratio(t, text) >= 85)
    return hits / len(tokens)


# ------------------------------------------------------------- verification --

def verify_packaged(ocr_text: str, nutrients: dict[str, NutrientValue], product_name: str | None,
                    user_barcode: str | None, detected_barcode: str | None, thresholds: dict) -> dict:
    guard = thresholds["guardrails"]
    product_name = (product_name or "").strip()
    user_barcode = re.sub(r"\D", "", user_barcode or "") or None
    checks, status = [], None

    if not product_name and not user_barcode and not detected_barcode:
        return {"status": "SKIPPED", "message": "No product name or barcode given - identity not checked.",
                "checks": [], "reference": None}

    if user_barcode and detected_barcode and user_barcode != detected_barcode:
        status = "MISMATCH"
        checks.append({"check": "barcode", "ok": False,
                       "detail": f"Barcode on the pack ({detected_barcode}) differs from the one entered "
                                 f"({user_barcode})."})

    code = user_barcode or detected_barcode
    reference, ref_via = None, None
    if code:
        reference, ref_via = off_by_barcode(code), f"barcode {code}"
    if reference is None and product_name:
        reference, ref_via = off_search(product_name), f"name search '{product_name}'"
    if reference:
        checks.append({"check": "reference", "ok": True,
                       "detail": f"Open Food Facts: {reference.get('brands', '?')} - "
                                 f"{reference.get('product_name', '?')} (via {ref_via})."})
    else:
        checks.append({"check": "reference", "ok": None,
                       "detail": "No Open Food Facts record found (or offline)."})

    # Brand / name check against what is printed on the pack.
    claimed = " ".join(filter(None, [product_name,
                                     (reference or {}).get("brands", ""),
                                     (reference or {}).get("product_name", "") if not product_name else ""]))
    claimed_families = set(find_brands(claimed).values())
    printed = find_brands(ocr_text)
    foreign = {b: f for b, f in printed.items() if claimed_families and f not in claimed_families}
    visible = name_visible(claimed, ocr_text) if claimed else 0.0

    if foreign and not set(printed.values()) & claimed_families:
        status = "MISMATCH"
        checks.append({"check": "brand", "ok": False,
                       "detail": f"The pack shows {', '.join(sorted(foreign))}, which is not the product named "
                                 f"('{claimed.strip()}')."})
    elif claimed and (set(printed.values()) & claimed_families or visible >= 0.5):
        checks.append({"check": "brand", "ok": True,
                       "detail": f"Product name/brand found on the pack ({visible:.0%} of name words matched)."})
    elif claimed:
        checks.append({"check": "brand", "ok": None,
                       "detail": "The claimed brand/name is not visible in this photo (back-of-pack photos "
                                 "often omit it)."})

    # Nutrient cross-check against the reference record.
    divergent = []
    ref_values = reference_nutrients(reference) if reference else {}
    for key, ref in ref_values.items():
        if key not in nutrients:
            continue
        got = nutrients[key].value
        abs_diff = abs(got - ref)
        rel_diff = abs_diff / max(ref, 1e-6)
        if abs_diff > guard["nutrient_divergence_abs_g"].get(key, 1.0) and rel_diff > guard["nutrient_divergence_rel"]:
            divergent.append(f"{DISPLAY_NAMES[key]}: label {got:g} g vs reference {ref:g} g")
    compared = sum(1 for k in ref_values if k in nutrients)
    if divergent:
        checks.append({"check": "nutrients", "ok": False, "detail": "; ".join(divergent)})
        if len(divergent) >= 3 and len(divergent) >= 0.6 * compared:
            status = "MISMATCH"          # nearly every value disagrees: a different product's label
        elif len(divergent) >= 2 and status != "MISMATCH":
            status = "DIVERGENT"
    elif ref_values and any(k in nutrients for k in ref_values):
        checks.append({"check": "nutrients", "ok": True, "detail": "Label values agree with the reference record."})

    if status is None:
        status = "VERIFIED" if any(c["ok"] for c in checks if c["check"] in ("brand", "nutrients")) else "UNVERIFIED"

    messages = {
        "VERIFIED": "Photo matches the named product.",
        "UNVERIFIED": "Could not confirm the product identity from this photo.",
        "DIVERGENT": "Warning: label values do not match the reference database for this product. "
                     "Make sure you uploaded the correct back-of-pack label.",
        "MISMATCH": "This image does not belong to the product you named / scanned.",
    }
    return {
        "status": status,
        "message": messages[status],
        "checks": checks,
        "barcode": code,
        "reference": {
            "product_name": reference.get("product_name"),
            "brands": reference.get("brands"),
            "image_url": reference.get("image_front_url"),
            "nutrients_per_100": ref_values,
        } if reference else None,
    }


def resolve_produce_name(name: str) -> str | None:
    norm = _norm(name)
    best, best_score = None, 0
    for key, aliases in PRODUCE_ALIASES.items():
        for alias in aliases:
            score = fuzz.WRatio(norm, alias)
            if score > best_score:
                best, best_score = key, score
    return best if best_score >= 85 else None


def verify_produce(product_name: str | None, clip_identity: dict | None, cnn_type: str | None) -> dict:
    product_name = (product_name or "").strip()
    if not product_name:
        return {"status": "SKIPPED", "message": "No produce name given - identity not checked.", "checks": []}
    claimed = resolve_produce_name(product_name)
    if claimed is None:
        return {"status": "UNVERIFIED", "message": f"'{product_name}' is not a produce name I recognise.",
                "checks": []}
    seen = {clip_identity["top"]} if clip_identity else set()
    if cnn_type:
        seen.add(cnn_type)
    pretty = lambda k: k.replace("_", " ")  # noqa: E731
    if claimed in seen:
        return {"status": "VERIFIED", "message": f"Image shows {pretty(claimed)}, as named.", "checks": []}
    # The CNN only knows supported types; trust CLIP's name only when it agrees it is one of them.
    shown = clip_identity["top"] if clip_identity and clip_identity["supported"] else cnn_type
    return {"status": "MISMATCH",
            "message": f"You named '{product_name}' ({pretty(claimed)}) but the image shows {pretty(shown)}.",
            "checks": []}
