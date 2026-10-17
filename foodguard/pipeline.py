"""FoodGuard AI orchestrator: guardrails -> category pipeline -> one clear verdict.

    Stage 1  Out-of-food filter        (CLIP zero-shot)         -> REJECTED
    Stage 2  Category validation       (CLIP + OCR text cues)   -> MISMATCH
    Stage 3  Product identity          (barcode / OFF / brands) -> MISMATCH
    Pipeline A  raw produce  -> MobileNetV2 spoilage           -> SAFE / CAUTION / UNSAFE
    Pipeline B  packaged     -> EasyOCR + FSSAI rules (+ optional expiry date) -> SAFE / CAUTION / UNSAFE
    Review      every analysed item gets a 0-100 health score, rating and food info (review.py)

Verdict meanings
    SAFE      fresh produce, or packaged food within HFSS limits (and in date, if a date was given)
    CAUTION   borderline produce, or a nutrient value that could not be read from the label
    UNSAFE    rotten produce, any FSSAI HFSS limit exceeded (red hexagons), or a user-entered
              expiry date that has passed
    REJECTED  image is not a usable food photo
    MISMATCH  image does not match the selected category or the named product
"""
import time
from datetime import date

from PIL import Image

from . import config, guardrails, identity, label_parser, ocr_engine, review
from .fssai_engine import check_expiry, evaluate_compliance, suggest_exempt
from .models import ModelNotTrainedError

HEADLINES = {
    "SAFE": "SAFE TO CONSUME",
    "CAUTION": "CONSUME WITH CAUTION",
    "UNSAFE": "UNSAFE TO CONSUME",
    "REJECTED": "IMAGE NOT COMPATIBLE",
    "MISMATCH": "MISMATCH DETECTED",
}


class Report:
    """Collects stage results and builds the JSON response."""

    def __init__(self, category: str):
        self.started = time.perf_counter()
        self.data = {
            "category": category,
            "category_label": config.CATEGORIES.get(category, category),
            "stages": [],
            "warnings": [],
            "hexagons": [],
            "evidence": [],
            "produce": None,
            "packaged": None,
            "identity": None,
            "suggested_category": None,
            "review": None,
        }

    def stage(self, name: str, status: str, detail: str, **extra):
        self.data["stages"].append({"name": name, "status": status, "detail": detail, **extra})

    def warn(self, message: str):
        if message not in self.data["warnings"]:
            self.data["warnings"].append(message)

    def finish(self, status: str, reason: str, details: list[str] | None = None) -> dict:
        self.data["verdict"] = {"status": status, "headline": HEADLINES[status],
                                "reason": reason, "details": details or []}
        self.data["elapsed_ms"] = round(1000 * (time.perf_counter() - self.started))
        return self.data


def analyse(images: list[Image.Image], category: str, product_name: str | None = None,
            barcode: str | None = None, expiry_date: date | None = None) -> dict:
    if category not in config.CATEGORIES:
        raise ValueError(f"Unknown category '{category}'")
    if not images:
        raise ValueError("At least one image is required")

    thresholds = config.load_thresholds()
    guard = thresholds["guardrails"]
    report = Report(category)

    # ---- Stage 1: is every image a food item? --------------------------------
    scenes = [guardrails.analyse_scene(img) for img in images]
    for i, scene in enumerate(scenes):
        label = f"Image {i + 1}" if len(images) > 1 else "Image"
        if scene["food_probability"] < guard["food_probability_min"]:
            report.stage("Stage 1 - Food filter", "FAIL",
                         f"{label} does not look like food (food probability "
                         f"{scene['food_probability']:.0%}; closest match: '{scene['best_description']}').")
            return report.finish("REJECTED", "This doesn't look like food. Try a clear photo of a fruit, a "
                                             "vegetable or a food pack.")
        if scene["visual_category"] == "prepared_food":
            report.stage("Stage 1 - Food filter", "FAIL",
                         f"{label} looks like cooked / prepared food ('{scene['best_description']}').")
            return report.finish("REJECTED", "This looks like a cooked dish. FoodGuard checks raw fruit and "
                                             "vegetables, and the labels of packaged food.")
    report.stage("Stage 1 - Food filter", "PASS",
                 f"Food item confirmed (probability {min(s['food_probability'] for s in scenes):.0%}).")

    if category == "raw_produce":
        return _raw_produce(images[0], scenes[0], product_name, report, guard, extra=len(images) - 1)
    return _packaged(images, scenes, category, product_name, barcode, expiry_date, report, thresholds)


# ------------------------------------------------------------- pipeline A ----

def _raw_produce(image, scene, product_name, report, guard, extra: int) -> dict:
    if extra:
        report.warn(f"Raw produce uses one photo; {extra} extra image(s) ignored.")

    if scene["packaged_probability"] > scene["raw_probability"] + guard["category_margin"]:
        report.stage("Stage 2 - Category validation", "FAIL",
                     f"Selected 'raw produce' but the image looks like packaged food "
                     f"(packaged {scene['packaged_probability']:.0%} vs raw {scene['raw_probability']:.0%}).")
        report.data["suggested_category"] = scene["visual_category"]
        return report.finish("MISMATCH", "This looks like a packaged product, not a fresh fruit or vegetable. "
                                         "Pick a packaged category and photograph the back of the pack.")

    ident = scene.get("produce_identity")
    if ident and not ident["supported"] and ident["supported_probability"] < guard["produce_supported_min_prob"]:
        supported = ", ".join(p.replace("_", " ") for p in config.SUPPORTED_PRODUCE)
        report.stage("Stage 2 - Category validation", "FAIL",
                     f"Produce identified as {ident['top'].replace('_', ' ')} ({ident['probability']:.0%}), "
                     f"which the spoilage model was not trained on.")
        return report.finish("REJECTED", f"This looks like {ident['top'].replace('_', ' ')}, which FoodGuard "
                                         f"has not learned to check yet. It can check {supported}.")
    report.stage("Stage 2 - Category validation", "PASS",
                 f"Raw produce confirmed"
                 + (f" - looks like {ident['top'].replace('_', ' ')} ({ident['probability']:.0%})." if ident else "."))

    from .vision import predict_spoilage
    try:
        result = predict_spoilage(image)
    except ModelNotTrainedError as exc:
        report.stage("Pipeline A - Spoilage CNN", "FAIL", str(exc))
        return report.finish("REJECTED", str(exc))

    if ident and ident["supported"] and ident["top"] != result["produce_type"]:
        report.warn(f"Vision models disagree on the produce type (CLIP: {ident['top'].replace('_', ' ')}, "
                    f"CNN: {result['produce_type'].replace('_', ' ')}). Retake the photo with one item, "
                    f"well lit, filling the frame.")

    check = identity.verify_produce(product_name, ident, result["produce_type"])
    report.data["identity"] = check
    report.stage("Stage 3 - Identity", {"VERIFIED": "PASS", "MISMATCH": "FAIL"}.get(check["status"], "SKIP"),
                 check["message"])
    if check["status"] == "MISMATCH":
        return report.finish("MISMATCH", check["message"])

    heatmap = result.pop("heatmap_png")
    report.data["produce"] = result
    report.data["review"] = review.for_produce(result)
    report.data["evidence"].append({"title": "Grad-CAM: regions that drove the decision", "png": heatmap})
    report.stage("Pipeline A - Spoilage CNN", "PASS",
                 f"P(fresh) {result['fresh_probability']:.0%}, P(stale) {result['stale_probability']:.0%}.")

    pretty = result["produce_type"].replace("_", " ")
    details = [f"Freshness confidence: {result['fresh_probability']:.1%}",
               f"Spoilage confidence: {result['stale_probability']:.1%}"]
    surface = result["surface_analysis"]
    if surface.get("foreground_found"):
        details.append(f"Dark / brown surface patches: {surface['dark_patch_percent']}% of the visible surface")
    reasons = {
        "SAFE": f"This {pretty} looks fresh (grade A). {result['action']}",
        "CAUTION": f"This {pretty} is past its best (grade B). {result['action']}",
        "UNSAFE": f"This {pretty} has gone bad (grade D). {result['action']}",
    }
    return report.finish(result["status"], reasons[result["status"]], details)


# ------------------------------------------------------------- pipeline B ----

OCR_VARIANTS = [                     # (name, preprocess key, EasyOCR options)
    ("colour", "color", {}),
    ("enhanced", "enhanced", {}),
    ("upscaled", "gray", {"mag_ratio": 1.5, "canvas_size": 3200}),
    ("binarised", "binary", {}),
]
GOOD_ENOUGH_NUTRIENTS = 5


def _ocr_image(image: Image.Image) -> dict:
    """OCR several pre-processed variants and merge them.

    The colour and CLAHE-enhanced variants always run; the slower up-scaled and
    binarised ones only when the table is still thin. The variant with the most
    nutrient rows is the primary reading. For each nutrient, a value that two
    variants agree on wins (simple majority vote against OCR noise); otherwise the
    primary reading is used, and nutrients it missed are filled from the others.
    """
    rules = config.load_thresholds()["nutrient_rules"]
    prep = ocr_engine.preprocess(image)
    runs = []
    for i, (name, key, options) in enumerate(OCR_VARIANTS):
        if i >= 2 and max(len(r["parsed"]["nutrients"]) for r in runs) >= GOOD_ENOUGH_NUTRIENTS:
            break
        boxes = ocr_engine.read_boxes(prep[key], **options)
        lines = [ln.text for ln in ocr_engine.group_lines(boxes)]
        runs.append({"variant": name, "boxes": boxes, "lines": lines,
                     "parsed": label_parser.parse_nutrients(lines, rules)})

    primary = max(runs, key=lambda r: (len(r["parsed"]["nutrients"]), -runs.index(r)))
    merged = dict(primary["parsed"]["nutrients"])
    for key in {k for r in runs for k in r["parsed"]["nutrients"]}:
        readings = [r["parsed"]["nutrients"][key] for r in runs if key in r["parsed"]["nutrients"]]
        # "33" in one variant and "3.3" in another: OCR drops decimal points far more often than
        # it invents them, so the reading with the smaller value by a factor of 10/100 wins.
        smaller = [c for c in readings for o in readings
                   if c.value > 0 and any(abs(o.value - c.value * f) <= 0.01 * o.value for f in (10, 100))]
        if smaller:
            merged[key] = smaller[0]
            continue
        for cand in readings:
            agree = sum(1 for o in readings if abs(o.value - cand.value) <= 0.01 * max(cand.value, 0.01))
            if agree >= 2:
                merged[key] = cand
                break
        else:
            merged.setdefault(key, readings[0])
    if any(merged[k] is not primary["parsed"]["nutrients"].get(k) for k in merged):
        # Values taken from another variant point at that variant's line numbers; don't highlight them.
        for k, nv in merged.items():
            if nv is not primary["parsed"]["nutrients"].get(k):
                merged[k] = label_parser.NutrientValue(nv.key, nv.value, nv.raw_text, -1, nv.less_than,
                                                       nv.source, nv.notes)
    parsed = {**primary["parsed"], "nutrients": merged}
    return {"prep": prep, "boxes": primary["boxes"], "lines": primary["lines"], "parsed": parsed,
            "variant": primary["variant"], "all_lines": [ln for r in runs for ln in r["lines"]]}


def _packaged(images, scenes, category, product_name, barcode, expiry_date, report, thresholds) -> dict:
    guard = thresholds["guardrails"]

    # ---- Stage 2a: visual check - is this actually packaged food? ------------
    if all(s["raw_probability"] > s["packaged_probability"] + guard["category_margin"] for s in scenes):
        report.stage("Stage 2 - Category validation", "FAIL",
                     f"Selected '{config.CATEGORIES[category]}' but the image shows raw produce "
                     f"(raw {scenes[0]['raw_probability']:.0%}).")
        report.data["suggested_category"] = "raw_produce"
        return report.finish("MISMATCH", "This is a fresh fruit or vegetable, not a food label. "
                                         "Pick 'Fruit or vegetable' instead.")

    # ---- OCR every image ------------------------------------------------------
    ocr = [_ocr_image(img) for img in images]
    all_lines = [line for o in ocr for line in o["lines"]]
    text_lines = [line for o in ocr for line in o["all_lines"]]     # every variant: more chances to read brands
    all_text = "\n".join(text_lines)
    total_boxes = sum(len(o["boxes"]) for o in ocr)

    nutrients: dict = {}
    parse_warnings = []
    layout_basis = None
    for o in ocr:
        for key, nv in o["parsed"]["nutrients"].items():
            nutrients.setdefault(key, nv)
        parse_warnings += o["parsed"]["warnings"]
        layout_basis = layout_basis or o["parsed"]["layout"].basis
    for w in parse_warnings:
        report.warn(w)
    hfss_keys = {"added_sugar_g", "total_sugar_g", "saturated_fat_g", "total_fat_g", "sodium_g"}
    found_hfss = hfss_keys & set(nutrients)

    # ---- Stage 2b: text checks - is there a readable label / nutrition panel? -
    if total_boxes < guard["min_text_boxes_packaged"]:
        report.stage("Stage 2 - Category validation", "FAIL",
                     f"Only {total_boxes} text regions detected - no readable label.")
        return report.finish("REJECTED", "We couldn't read any text on this pack. Take a sharp, well-lit photo "
                                         "of the back of the pack, without glare.")
    if category != "packaged_exempt" and len(nutrients) < guard["min_nutrition_keywords"]:
        report.stage("Stage 2 - Category validation", "FAIL",
                     f"{total_boxes} text regions read, but no nutrition table rows were recognised.")
        return report.finish("REJECTED", "We couldn't find the nutrition table. Photograph the part of the "
                                         "pack where the nutrition values are printed.")

    # Solid vs liquid from the printed basis ("per 100 g" / "per 100 ml").
    if layout_basis == "ml" and category == "packaged_solid":
        report.stage("Stage 2 - Category validation", "FAIL", "Label declares values per 100 ml (a beverage).")
        report.data["suggested_category"] = "packaged_liquid"
        return report.finish("MISMATCH", "The label gives values per 100 ml, so this is a drink. "
                                         "Pick 'Packaged drink' instead.")
    if layout_basis == "g" and category == "packaged_liquid":
        report.stage("Stage 2 - Category validation", "FAIL", "Label declares values per 100 g (a solid food).")
        report.data["suggested_category"] = "packaged_solid"
        return report.finish("MISMATCH", "The label gives values per 100 g, so this is a solid food. "
                                         "Pick 'Packaged food' instead.")
    scene = scenes[0]["group_probabilities"]
    if layout_basis is None and category == "packaged_solid" and \
            scene["packaged_liquid"] > scene["packaged_solid"] + guard["liquid_margin"]:
        report.warn("The pack looks like a beverage but the label basis (g / ml) was not readable - "
                    "check the selected category.")
    report.stage("Stage 2 - Category validation", "PASS",
                 f"Packaged label confirmed: {total_boxes} text regions, {len(nutrients)} nutrition rows"
                 + (f", basis per 100 {layout_basis}." if layout_basis else "."))

    # Exempt suggestion (single-ingredient products).
    prominent = [t for o in ocr for t in ocr_engine.prominent_text(o["boxes"])]
    ingredients = label_parser.parse_ingredients(all_lines)
    exempt_kw = suggest_exempt(prominent, ingredients, thresholds["exempt_keywords"])
    if exempt_kw and category != "packaged_exempt":
        report.data["suggested_category"] = "packaged_exempt"
        report.warn(f"This looks like a single-ingredient product ('{exempt_kw}'). FSSAI HFSS warnings may not "
                    f"apply - re-run as 'Packaged - Exempt' if so.")

    # ---- Stage 3: identity ----------------------------------------------------
    detected_code = next((c for c in (identity.decode_barcode(o["prep"]["color"]) for o in ocr) if c), None)
    check = identity.verify_packaged(all_text, nutrients, product_name, barcode, detected_code, thresholds)
    report.data["identity"] = check
    report.stage("Stage 3 - Identity",
                 {"VERIFIED": "PASS", "MISMATCH": "FAIL", "DIVERGENT": "WARN"}.get(check["status"], "SKIP"),
                 check["message"], checks=check["checks"])
    if check["status"] == "MISMATCH":
        return report.finish("MISMATCH", check["message"],
                             [c["detail"] for c in check["checks"] if c["ok"] is False])
    if check["status"] == "DIVERGENT":
        report.warn(check["message"])

    # Fill nutrients the OCR missed from a barcode-verified reference record.
    ref = (check.get("reference") or {}).get("nutrients_per_100", {})
    if check["status"] == "VERIFIED" and check.get("barcode"):
        for key, value in ref.items():
            if key not in nutrients:
                nutrients[key] = label_parser.NutrientValue(key, value, "Open Food Facts record", -1,
                                                            source="Open Food Facts")

    # ---- FSSAI compliance + optional expiry -----------------------------------
    compliance = evaluate_compliance(nutrients, category, thresholds)
    expiry = check_expiry(expiry_date)
    report.data["hexagons"] = compliance["hexagons"]

    first = ocr[0]
    highlight_lines = {nv.line_index for nv in first["parsed"]["nutrients"].values()}
    box_index = {id(b): i for i, b in enumerate(first["boxes"])}
    grouped = ocr_engine.group_lines(first["boxes"])
    highlight_boxes = {box_index[id(b)] for i, ln in enumerate(grouped) if i in highlight_lines
                       for b in ln.boxes}
    report.data["evidence"].append({"title": "OCR text regions (nutrition rows in red)",
                                    "png": ocr_engine.annotate(first["prep"]["color"], first["boxes"],
                                                               highlight_boxes)})
    report.data["packaged"] = {
        "nutrients": {k: v.to_dict() for k, v in nutrients.items()},
        "compliance": compliance,
        "expiry": expiry,
        "basis": layout_basis,
        "ingredients": ingredients,
        "prominent_text": prominent[:6],
        "ocr_variant": [o["variant"] for o in ocr],
        "ocr_text": all_lines,
    }
    report.data["review"] = review.for_packaged(nutrients, compliance, expiry, category, product_name, check,
                                              ingredients, layout_basis)
    report.stage("Pipeline B - OCR + FSSAI engine", "PASS",
                 f"{len(nutrients)} nutrients parsed"
                 + (f"; expiry {expiry['status'].lower()}." if expiry["expiry_date"] else "; no expiry date entered."))

    return _packaged_verdict(report, compliance, expiry, category)


def _packaged_verdict(report, compliance, expiry, category) -> dict:
    details = []
    for c in compliance["checks"]:
        if c["value"] is None:
            details.append(f"{c['name']}: not readable (limit {c['limit']:g} g)")
        else:
            details.append(f"{c['name']}: {'<' if c['less_than'] else ''}{c['value']:g} g per 100 "
                           f"{compliance['basis'][3:]} (limit {c['limit']:g} g) - {c['status'].replace('_', ' ')}")
    if expiry["expiry_date"]:
        details.append(f"Expiry: {expiry['expiry_date']} ({expiry['days_left']} days left)"
                       if expiry["days_left"] >= 0 else
                       f"Expired on {expiry['expiry_date']} ({-expiry['days_left']} days ago)")

    if expiry["status"] == "EXPIRED":
        when = date.fromisoformat(expiry["expiry_date"])
        return report.finish("UNSAFE", f"This pack expired on {when.day} {when:%B %Y}. Throw it away.", details)
    if compliance["hexagons"]:
        over = _and_list([c["name"].lower() for c in compliance["checks"] if c["status"] == "EXCEEDED"])
        return report.finish("UNSAFE", f"It has too much {over} for its category, so FSSAI rules call for "
                                       f"the red warning marks below.", details)

    unverified = [c["name"].lower() for c in compliance["checks"] if c["status"] in ("NOT_FOUND", "UNCERTAIN")]
    if unverified:
        return report.finish("CAUTION", f"We couldn't read the {_and_list(unverified)} on the label. A clearer "
                                        f"photo of the nutrition table would let us finish the check.", details)
    in_date = "It is in date. " if expiry["status"] == "VALID" else ""
    if category == "packaged_exempt":
        return report.finish("SAFE", f"{in_date}This is a single-ingredient food, so the FSSAI warning limits "
                                     f"don't apply.", details)
    checked = _and_list([c["name"].lower() for c in compliance["checks"]])
    return report.finish("SAFE", f"{in_date}{checked.capitalize()} {'are' if len(compliance['checks']) > 1 else 'is'} "
                                 f"within the FSSAI limits.", details)


def _and_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
