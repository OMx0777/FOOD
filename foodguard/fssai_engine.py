"""Pipeline B, step 3: FSSAI HFSS (High Fat, Sugar, Salt) compliance rules.

Limits are category-specific (solid per 100 g vs liquid per 100 ml) and read
from fssai_config/safety_thresholds.json. Single-ingredient foods (oil, ghee,
honey, dried fruit ...) are exempt from HFSS warnings.

The expiry check is optional: it runs only when the user types the date printed
on the pack. A missing date never lowers the verdict.
"""
import re
from datetime import date

from .label_parser import DISPLAY_NAMES, NutrientValue

HEXAGON_LABELS = {
    "added_sugar_g": "HIGH SUGAR",
    "saturated_fat_g": "HIGH SAT FAT",
    "sodium_g": "HIGH SALT",
}
# Words that mean "this is a processed snack", so a keyword like "butter" in
# "Butter Cookies" must not trigger the exempt suggestion.
PROCESSED_WORDS = ["cookie", "biscuit", "chips", "cake", "bread", "popcorn", "chocolate", "candy",
                   "namkeen", "masala", "noodles", "bhujia", "crackers", "toffee", "spread", "flavour"]


def evaluate_compliance(nutrients: dict[str, NutrientValue], category: str, thresholds: dict) -> dict:
    cat = thresholds["packaged_categories"][category]
    rules = thresholds["nutrient_rules"]
    if cat.get("hfss_exempt"):
        return {"exempt": True, "basis": cat["basis"], "checks": [], "hexagons": [], "missing": []}

    checks, hexagons, missing = [], [], []
    for key, limit in cat["limits"].items():
        nv = nutrients.get(key)
        note = None
        if nv is None and key == "added_sugar_g" and rules.get("use_total_sugar_if_added_missing"):
            nv = nutrients.get("total_sugar_g")
            if nv is not None:
                note = "Added sugar not declared - total sugars used (conservative)."
        if nv is None:
            missing.append(key)
            checks.append({"key": key, "name": DISPLAY_NAMES[key], "limit": limit, "value": None,
                           "status": "NOT_FOUND", "note": "Not readable on the label photo."})
            continue

        if nv.less_than and nv.value <= limit:
            status = "OK"
        elif nv.less_than:
            status = "UNCERTAIN"
            note = f"Label says '< {nv.value:g}', which cannot be compared with the {limit:g} limit."
        else:
            status = "EXCEEDED" if nv.value > limit else "OK"

        check = {
            "key": key, "name": DISPLAY_NAMES[key], "limit": limit, "value": round(nv.value, 3),
            "less_than": nv.less_than, "status": status, "source": nv.source,
            "raw_text": nv.raw_text, "note": note or "; ".join(nv.notes) or None,
            "percent_of_limit": round(100 * nv.value / limit) if limit else None,
        }
        checks.append(check)
        if status == "EXCEEDED":
            hexagons.append({"key": key, "label": HEXAGON_LABELS[key],
                             "value": round(nv.value, 3), "limit": limit, "unit": "g"})
        elif status == "UNCERTAIN":
            missing.append(key)

    return {"exempt": False, "basis": cat["basis"], "checks": checks, "hexagons": hexagons, "missing": missing}


def check_expiry(expiry_date: date | None, today: date | None = None) -> dict:
    """Optional, user-entered expiry / best-before date."""
    if expiry_date is None:
        return {"status": "NOT_PROVIDED", "expiry_date": None, "days_left": None}
    days_left = (expiry_date - (today or date.today())).days
    return {"status": "EXPIRED" if days_left < 0 else "VALID",
            "expiry_date": expiry_date.isoformat(), "days_left": days_left}


def suggest_exempt(prominent: list[str], ingredients: list[str], keywords: list[str]) -> str | None:
    """Return the exempt keyword when the pack looks like a single-ingredient product."""
    title = " ".join(prominent).lower()
    if any(w in title for w in PROCESSED_WORDS):
        return None
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", title):
            return kw
    if 0 < len(ingredients) <= 2:
        joined = " ".join(ingredients).lower()
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", joined):
                return kw
    return None
