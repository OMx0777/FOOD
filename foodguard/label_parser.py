"""Pipeline B, step 2: turn OCR text lines into structured nutrition data.

Pure functions over a list of text lines (no OCR / model imports), so every
rule here is unit-tested in tests/test_label_parser.py.

Nutrients
---------
* Each line is split into a *label* (text before the first number) and *values*.
* The label is fuzzy-matched (RapidFuzz - the maintained, MIT-licensed
  implementation of the FuzzyWuzzy API) against nutrient aliases, so OCR slips
  like "Saturated Fal" or "Sodiurn" still match.
* Column handling: FSSAI labels must declare values per 100 g / 100 ml and may
  add a per-serve and %RDA column. The header row tells us the column order; a
  per-serve-only label is normalised with the declared serving size.
* Units are normalised to grams (mg -> g); salt is converted to sodium (/2.5).
"""
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

# ---------------------------------------------------------------- nutrients --

NUTRIENT_ALIASES = {
    "added_sugar_g": ["added sugar", "added sugars", "incl added sugars", "includes added sugars",
                      "of which added sugars"],
    "total_sugar_g": ["total sugar", "total sugars", "sugar", "sugars", "of which sugars"],
    "saturated_fat_g": ["saturated fat", "saturated fats", "saturated fatty acids", "sat fat",
                        "saturates", "saturated", "of which saturated fat", "sfa"],
    "trans_fat_g": ["trans fat", "trans fats", "trans fatty acids", "tfa"],
    "total_fat_g": ["total fat", "fat", "fats", "total fats"],
    "sodium_g": ["sodium"],
    "salt_g": ["salt"],
    "energy_kcal": ["energy", "calories", "energy value"],
    "protein_g": ["protein", "proteins"],
    "carbohydrate_g": ["carbohydrate", "carbohydrates", "total carbohydrate", "total carbohydrates"],
}
DISPLAY_NAMES = {
    "added_sugar_g": "Added sugar",
    "total_sugar_g": "Total sugars",
    "saturated_fat_g": "Saturated fat",
    "trans_fat_g": "Trans fat",
    "total_fat_g": "Total fat",
    "sodium_g": "Sodium",
    "salt_g": "Salt",
    "energy_kcal": "Energy",
    "protein_g": "Protein",
    "carbohydrate_g": "Carbohydrate",
}
LABEL_MATCH_MIN = 82

NUMBER = r"\d+(?:[.,]\d+)?"
VALUE_RE = re.compile(rf"(<|less than\s*)?\s*({NUMBER})\s*(mg|mcg|µg|ug|g|kcal|kj|%)?(?![a-z])", re.I)
SERVING_RE = re.compile(rf"serv\w*\s*(?:size)?\s*(?:is)?\s*[:\-=]?\s*\(?\s*({NUMBER})\s*(g|ml|9)\b", re.I)
SERVING_BEFORE_RE = re.compile(rf"per\s*({NUMBER})\s*(g|ml|9)\s*serv", re.I)     # "Per 30 g serving"


@dataclass
class NutrientValue:
    key: str
    value: float                # grams per 100 g / 100 ml (kcal for energy)
    raw_text: str
    line_index: int
    less_than: bool = False
    source: str = "ocr"
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": DISPLAY_NAMES.get(self.key, self.key),
            "value": round(self.value, 3),
            "less_than": self.less_than,
            "raw_text": self.raw_text,
            "source": self.source,
            "notes": self.notes,
        }


def _clean_ocr_digits(text: str) -> str:
    """Fix the usual OCR digit confusions, but only inside number-like tokens."""
    def fix(tok: re.Match) -> str:
        t = tok.group(0)
        digits = sum(c.isdigit() for c in t)
        punct = sum(c in ".," for c in t)
        if digits == 0 or digits + punct < len(t) / 2:
            return t
        return t.translate(str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1", "S": "5"}))
    text = re.sub(r"[0-9OoDlI|S.,]+", fix, text)
    text = re.sub(r"(\d)\s*m\s*[q9]\b", r"\1mg", text)    # "520 mq" / "83m9" -> mg
    text = re.sub(r"(\d)\s*q\b", r"\1g", text)            # "3.2q"  -> 3.2g
    text = re.sub(r"(\d)e\b", r"\1g", text)                # "0.78e" -> 0.78g
    return text


def _normalise_label(text: str) -> str:
    text = re.sub(r"\(.*?\)", " ", text.lower())       # "(g)", "(mg)"
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\b(per|g|mg|kcal|kj)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_nutrient(label: str) -> tuple[str | None, float]:
    norm = _normalise_label(label)
    if not norm or len(norm) > 40:              # long phrases are sentences / ingredient lists
        return None, 0.0
    variants = {norm, norm.replace("rn", "m")}         # OCR commonly reads "m" as "rn"
    best_key, best_score = None, 0.0
    for key, aliases in NUTRIENT_ALIASES.items():
        for alias in aliases:
            score = max(fuzz.ratio(v, alias) for v in variants)
            # longer, more specific aliases win ties ("saturated fat" over "fat")
            score += len(alias) / 100
            if score > best_score:
                best_key, best_score = key, score
    return (best_key, best_score) if best_score >= LABEL_MATCH_MIN else (None, best_score)


@dataclass
class TableLayout:
    basis: str | None = None             # "g" or "ml" (from "per 100 g / ml")
    per100_column: int = 0               # index of the per-100 column among numeric columns
    has_per100: bool = False
    has_per_serving: bool = False
    serving_size: float | None = None
    serving_unit: str | None = None


def detect_layout(lines: list[str]) -> TableLayout:
    layout = TableLayout()
    for line in lines:
        low = _clean_ocr_digits(line).lower().replace(" ", "")
        m100 = re.search(r"100(ml|mi|m1|grams?|gms|gm|g)", low)      # spaces removed: no \b here
        # A per-serve *column* header - a bare "Serving size 30 g" line does not count, because
        # per-100 g tables print it too and OCR often misses the "per 100 g" header itself.
        mserv = re.search(r"(perserv|amountperserv|/serv|perpack|per\d+(?:\.\d+)?(?:g|ml|9)serv)", low)
        if m100:
            layout.has_per100 = True
            layout.basis = layout.basis or ("ml" if m100.group(1) in ("ml", "mi", "m1") else "g")
        if mserv:
            layout.has_per_serving = True
        if m100 and mserv and mserv.start() < m100.start():
            layout.per100_column = 1
    for line in lines:
        clean = _clean_ocr_digits(line)
        m = SERVING_RE.search(clean) or SERVING_BEFORE_RE.search(clean)
        if m:
            layout.serving_size = float(m.group(1).replace(",", "."))
            layout.serving_unit = "ml" if m.group(2).lower() == "ml" else "g"
            break
    return layout


def _to_grams(value: float, unit: str | None, key: str) -> tuple[float, str | None]:
    unit = (unit or "").lower()
    if key == "energy_kcal":
        return (value / 4.184 if unit == "kj" else value), None
    if unit == "mg":
        return value / 1000, None
    if unit in ("mcg", "µg", "ug"):
        return value / 1_000_000, None
    if not unit and key == "sodium_g" and value > 5:
        return value / 1000, "unit missing; value > 5 treated as mg"
    return value, None


def _number(token: str, unit: str) -> tuple[float, str]:
    """Parse one numeric token, repairing two frequent OCR slips:
    "078" -> 0.78 (dot after a leading zero lost) and "33.29" -> 33.2 g (unit g read as 9)."""
    if re.fullmatch(r"0\d+", token):
        return float(f"0.{token[1:]}"), unit
    if not unit and re.fullmatch(r"\d+[.,]\d+9", token):
        return float(token[:-1].replace(",", ".")), "g"
    return float(token.replace(",", ".")), unit


def _restore_decimal(per100: float, per_serve: float | None, layout: TableLayout) -> float:
    """OCR often drops the decimal point ("3.3" -> "33"). With a serving size we know that
    per_serve ~= per100 * serving / 100, so try per100, per100/10, per100/100 and keep the one
    that fits."""
    if per_serve is None or not layout.serving_size or per100 == 0:
        return per100
    ratio = layout.serving_size / 100
    for cand in (per100, per100 / 10, per100 / 100):
        if abs(cand * ratio - per_serve) <= 0.15 * max(per_serve, 0.05):
            return cand
    return per100


def parse_nutrients(lines: list[str], rules: dict | None = None) -> dict:
    """Return {"nutrients": {key: NutrientValue}, "layout": TableLayout, "warnings": [...]}."""
    rules = rules or {}
    layout = detect_layout(lines)
    found: dict[str, NutrientValue] = {}
    warnings: list[str] = []

    # Pass 1: read every nutrient row -> (value, unit, less_than, notes).
    rows: dict[str, tuple] = {}
    for idx, raw in enumerate(lines):
        line = _clean_ocr_digits(raw)
        line = re.sub(r"\(\s*[0689gq]\s*\)", "(g)", line)       # "(g)" unit read as "(0)" / "(9)"
        first_digit = re.search(r"[<\d]", line)
        if not first_digit:
            continue
        label = line[: first_digit.start()]
        key, _ = match_nutrient(label)
        if key is None or key in rows:
            continue
        label_unit = re.search(r"\(\s*(mg|g|kcal|kj)\b", label, re.I)   # "SODIUM (mg) 296"
        label_unit = label_unit.group(1).lower() if label_unit else ""

        values = []
        for m in VALUE_RE.finditer(line[first_digit.start():]):
            unit = (m.group(3) or label_unit).lower()
            if unit == "%":
                continue                                   # %RDA column
            if key == "energy_kcal" and unit == "kj":
                continue
            number, unit = _number(m.group(2), unit)
            values.append((number, unit, bool(m.group(1)), m.group(2)))
        if not values:
            continue

        # A "(g)" unit column is often read as a lone 9 / 8 / 6 before the real value:
        # "Saturated Fat 9 14.0". Drop it unless the table really has a per-serve column.
        if len(values) >= 2 and not layout.has_per_serving and not values[0][1] and values[0][3] in ("6", "8", "9"):
            values = values[1:]
        values = [v[:3] for v in values]

        notes = []
        if len(values) >= 2 and layout.has_per100:
            i100 = min(layout.per100_column, len(values) - 1)
            value, unit, lt = values[i100]
            restored = _restore_decimal(value, values[1 - i100][0] if i100 <= 1 else None, layout)
            if restored != value:
                notes.append(f"decimal point restored ({value:g} -> {restored:g}) using the per-serve column")
                value = restored
        else:
            value, unit, lt = values[0]
        second = values[1][0] if len(values) >= 2 else None
        rows[key] = (value, unit, lt, notes, raw.strip(), idx, second)

    # Pass 2: a per-serve-only table is scaled to per 100 - unless (a) the two value columns
    # relate as serving/100, proving column 1 is per 100 g and column 2 per serve, or (b) scaling
    # gives physically impossible numbers. Both mean OCR merely missed the "per 100 g" header.
    scale = 1.0
    if layout.has_per_serving and not layout.has_per100 and layout.serving_size:
        factor = 100 / layout.serving_size
        ratios = sorted(r[6] / r[0] for r in rows.values() if r[6] and r[0] > 0.5)
        first_is_per100 = bool(ratios) and abs(ratios[len(ratios) // 2] * factor - 1) <= 0.15
        impossible = first_is_per100 or any(
            (k == "energy_kcal" and _to_grams(v, u, k)[0] * factor > 900) or
            (k.endswith("_g") and k != "sodium_g" and _to_grams(v, u, k)[0] * factor > 100)
            for k, (v, u, *_rest) in rows.items())
        if impossible:
            warnings.append("Per-100 g header not read; the column ratios / plausible values show the first "
                            "column is per 100 g, so it is used unscaled.")
        else:
            scale = factor

    for key, (value, unit, lt, notes, raw_text, idx, _second) in rows.items():
        if scale != 1.0:
            value *= scale
            notes.append(f"converted from per-serving ({layout.serving_size:g}{layout.serving_unit})")
        elif layout.has_per_serving and not layout.has_per100 and not layout.serving_size:
            notes.append("per-serving column but serving size not found; first column used")
        grams, unit_note = _to_grams(value, unit, key)
        if unit_note:
            notes.append(unit_note)
        if not layout.has_per100 and not layout.has_per_serving:
            notes.append("basis not printed; assumed per 100")
        found[key] = NutrientValue(key, grams, raw_text, idx, less_than=lt, notes=notes)

    # Salt -> sodium when sodium itself is not declared.
    factor = rules.get("salt_to_sodium_factor", 2.5)
    if "sodium_g" not in found and "salt_g" in found:
        salt = found["salt_g"]
        found["sodium_g"] = NutrientValue("sodium_g", salt.value / factor, salt.raw_text, salt.line_index,
                                          salt.less_than, notes=[f"derived from salt / {factor}"])

    # Plausibility checks: drop impossible values, flag inconsistent ones.
    limits = rules.get("max_plausible_g_per_100", {})
    for key in list(found):
        if key in limits and found[key].value > limits[key]:
            warnings.append(f"{DISPLAY_NAMES[key]} value {found[key].value:g} is impossible per 100; ignored "
                            f"(OCR misread?): '{found[key].raw_text}'")
            del found[key]
    for part, whole in (("saturated_fat_g", "total_fat_g"), ("added_sugar_g", "total_sugar_g")):
        if part in found and whole in found and found[part].value > found[whole].value + 0.05:
            warnings.append(f"{DISPLAY_NAMES[part]} ({found[part].value:g}) exceeds "
                            f"{DISPLAY_NAMES[whole]} ({found[whole].value:g}) - check the label photo.")

    return {"nutrients": found, "layout": layout, "warnings": warnings}


# -------------------------------------------------------------- ingredients --

def parse_ingredients(lines: list[str]) -> list[str]:
    text = " ".join(lines)
    m = re.search(r"ingredients?\s*[:\-]?\s*(.+?)(?:nutrition|nutritional|allergen|contains|"
                  r"information|storage|best before|mfd|mrp|$)", text, re.I | re.S)
    if not m:
        return []
    body, depth, current, items = m.group(1), 0, "", []
    for ch in body:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch in ",;" and depth == 0:
            items.append(current)
            current = ""
        else:
            current += ch
    items.append(current)
    return [re.sub(r"\s+", " ", i).strip(" .") for i in items if i.strip(" .")]
