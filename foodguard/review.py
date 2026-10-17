"""Food review: a 0-100 health score with a rating, plus information about the food.

Raw produce
    score = freshness confidence of the CNN (P(fresh) x 100).
    info  = nutrition per 100 g, what the item is known for, storage and spoilage
            signs (foodguard/data/produce_info.json, USDA FoodData Central).

Packaged food
    score starts at 100 and loses points for each HFSS nutrient as it approaches and
    passes its FSSAI limit (see _nutrient_penalty). Any limit exceeded caps the score
    at 45 and an expired pack scores 0, so the score always agrees with the verdict.
    info  = product, nutrition per 100 g / ml and the share of a WHO daily maximum.
"""
import json
from functools import lru_cache

from . import config
from .label_parser import DISPLAY_NAMES

# WHO adult daily guidance (2,000 kcal diet): free sugars < 10 % energy, saturated fat
# < 10 % energy, sodium < 2 g.
DAILY_MAX = {"energy_kcal": 2000, "added_sugar_g": 50, "saturated_fat_g": 22, "sodium_g": 2.0}
EXCEEDED_CAP = 45
RATINGS = [(85, "Excellent"), (70, "Good"), (50, "Fair"), (30, "Poor"), (0, "Avoid")]
FALLBACK_TITLES = {"packaged_solid": "Your packaged food", "packaged_liquid": "Your packaged drink",
                   "packaged_exempt": "Your single-ingredient food"}


@lru_cache(maxsize=1)
def _produce_db() -> dict:
    return json.loads((config.ROOT / "foodguard" / "data" / "produce_info.json").read_text(encoding="utf-8"))


def _rating(score: int) -> tuple[str, float]:
    label = next(name for floor, name in RATINGS if score >= floor)
    return label, round(score / 20 * 2) / 2          # 0-5 stars in half steps


def _scored(score: float, breakdown: list[dict], info: dict) -> dict:
    score = int(round(max(0, min(100, score))))
    rating, stars = _rating(score)
    return {"score": score, "rating": rating, "stars": stars, "breakdown": breakdown, "info": info}


# ------------------------------------------------------------------- produce --

def for_produce(result: dict) -> dict:
    item = _produce_db().get(result["produce_type"], {})
    fresh = result["fresh_probability"]
    breakdown = [{"factor": "Freshness confidence (spoilage model)", "points": round(100 * fresh)}]
    surface = result.get("surface_analysis") or {}
    facts = [
        ("Type", f"{item.get('kind', 'Produce')}"),
        ("Energy", f"{item['energy_kcal']} kcal") if item.get("energy_kcal") is not None else None,
        ("Carbohydrate", f"{item['carbohydrate_g']} g") if item.get("carbohydrate_g") is not None else None,
        ("Dietary fibre", f"{item['fibre_g']} g") if item.get("fibre_g") is not None else None,
        ("Natural sugars", f"{item['sugar_g']} g") if item.get("sugar_g") is not None else None,
        ("Vitamin C", f"{item['vitamin_c_mg']} mg") if item.get("vitamin_c_mg") is not None else None,
        ("Potassium", f"{item['potassium_mg']} mg") if item.get("potassium_mg") is not None else None,
    ]
    info = {
        "title": item.get("name", result["produce_type"].replace("_", " ").title()),
        "subtitle": (f"Also called {item['local_name']}. " if item.get("local_name") else "") + "Values per 100 g, raw.",
        "facts": [{"label": k, "value": v} for k, v in filter(None, facts)],
        "known_for": item.get("known_for", []),
        "storage": item.get("storage"),
        "shelf_life": item.get("shelf_life"),
        "spoilage_signs": item.get("spoilage_signs"),
        "surface_dark_percent": surface.get("dark_patch_percent"),
        "source": _produce_db().get("_source"),
    }
    return _scored(100 * fresh, breakdown, info)


# ------------------------------------------------------------------ packaged --

def _nutrient_penalty(ratio: float) -> float:
    """0 up to half the limit, rising to 12 at the limit and to 35 at 3x the limit."""
    if ratio <= 0.5:
        return 0.0
    if ratio <= 1.0:
        return 12 * (ratio - 0.5) / 0.5
    return 12 + 23 * min(1.0, (ratio - 1.0) / 2.0)


def for_packaged(nutrients: dict, compliance: dict, expiry: dict, category: str,
                 product_name: str | None, identity: dict | None, ingredients: list[str], basis: str) -> dict:
    ref = (identity or {}).get("reference") or {}
    title = (product_name or "").strip() or " ".join(filter(None, [ref.get("brands"), ref.get("product_name")])) \
        or FALLBACK_TITLES[category]
    unit = "ml" if basis == "ml" or category == "packaged_liquid" else "g"

    facts = []
    for key in ("energy_kcal", "total_fat_g", "saturated_fat_g", "trans_fat_g", "carbohydrate_g",
                "total_sugar_g", "added_sugar_g", "protein_g", "sodium_g"):
        nv = nutrients.get(key)
        if nv is None:
            continue
        if key == "energy_kcal":
            value = f"{nv.value:.0f} kcal"
        elif key == "sodium_g":
            value = f"{nv.value * 1000:.0f} mg"
        else:
            value = f"{'<' if nv.less_than else ''}{nv.value:g} g"
        facts.append({"label": DISPLAY_NAMES[key], "value": value})

    daily = []
    for key, maximum in DAILY_MAX.items():
        nv = nutrients.get(key) or (nutrients.get("total_sugar_g") if key == "added_sugar_g" else None)
        if nv is None:
            continue
        daily.append({"name": DISPLAY_NAMES[key],
                      "percent": round(100 * nv.value / maximum),
                      "daily_max": f"{maximum:g} {'kcal' if key == 'energy_kcal' else 'g'}"})

    info = {
        "title": title,
        "subtitle": f"{config.CATEGORIES[category]}. Values per 100 {unit}.",
        "facts": facts,
        "daily_share": daily,
        "daily_note": f"Share of the WHO daily maximum for an adult, in 100 {unit}.",
        "ingredients": ingredients[:12],
        "ingredient_count": len(ingredients),
        "source": "Values read from the label photo" + (" and Open Food Facts" if ref else ""),
    }

    if expiry["status"] == "EXPIRED":
        return _scored(0, [{"factor": f"Expired on {expiry['expiry_date']}", "points": -100}], info)
    if compliance.get("exempt"):
        result = _scored(100, [], info)
        result.update(score=None, rating="Exempt", stars=None)
        result["breakdown"] = [{"factor": "Single-ingredient food - not scored against HFSS limits", "points": 0}]
        return result

    score, breakdown = 100.0, []
    for check in compliance["checks"]:
        if check["value"] is None:
            breakdown.append({"factor": f"{check['name']}: not readable, not scored", "points": 0})
            continue
        penalty = _nutrient_penalty(check["value"] / check["limit"])
        score -= penalty
        breakdown.append({"factor": f"{check['name']} at {check['value'] / check['limit']:.0%} of the FSSAI limit",
                          "points": -round(penalty)})
    if compliance["hexagons"]:
        if score > EXCEEDED_CAP:
            breakdown.append({"factor": "Capped: at least one FSSAI limit exceeded", "points": -round(score - EXCEEDED_CAP)})
        score = min(score, EXCEEDED_CAP)
    return _scored(score, breakdown, info)
