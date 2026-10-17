import pytest

from foodguard import identity
from foodguard.config import load_thresholds
from foodguard.fssai_engine import evaluate_compliance, suggest_exempt
from foodguard.label_parser import NutrientValue

TH = load_thresholds()


def nv(key, value, less_than=False):
    return NutrientValue(key, value, f"{key} {value}", 0, less_than=less_than)


# ------------------------------------------------------------------ FSSAI --

def test_solid_limits_raise_hexagons():
    nutrients = {"added_sugar_g": nv("added_sugar_g", 25.0), "saturated_fat_g": nv("saturated_fat_g", 6.0),
                 "sodium_g": nv("sodium_g", 0.296)}
    out = evaluate_compliance(nutrients, "packaged_solid", TH)
    assert {h["label"] for h in out["hexagons"]} == {"HIGH SUGAR", "HIGH SAT FAT"}
    assert not out["missing"]


def test_liquid_limits_are_stricter():
    nutrients = {"added_sugar_g": nv("added_sugar_g", 2.0), "saturated_fat_g": nv("saturated_fat_g", 0),
                 "sodium_g": nv("sodium_g", 0.01)}
    assert evaluate_compliance(nutrients, "packaged_solid", TH)["hexagons"] == []
    assert [h["label"] for h in evaluate_compliance(nutrients, "packaged_liquid", TH)["hexagons"]] == ["HIGH SUGAR"]


def test_exempt_category_has_no_hfss_checks():
    out = evaluate_compliance({"saturated_fat_g": nv("saturated_fat_g", 62.0)}, "packaged_exempt", TH)
    assert out["exempt"] and out["hexagons"] == []


def test_total_sugar_fallback_and_missing_values():
    out = evaluate_compliance({"total_sugar_g": nv("total_sugar_g", 9.0)}, "packaged_solid", TH)
    sugar = next(c for c in out["checks"] if c["key"] == "added_sugar_g")
    assert sugar["status"] == "EXCEEDED" and "total sugars" in sugar["note"]
    assert set(out["missing"]) == {"saturated_fat_g", "sodium_g"}


def test_less_than_values():
    out = evaluate_compliance({"added_sugar_g": nv("added_sugar_g", 0.5, less_than=True),
                               "saturated_fat_g": nv("saturated_fat_g", 5, less_than=True),
                               "sodium_g": nv("sodium_g", 0.1)}, "packaged_solid", TH)
    status = {c["key"]: c["status"] for c in out["checks"]}
    assert status == {"added_sugar_g": "OK", "saturated_fat_g": "UNCERTAIN", "sodium_g": "OK"}


@pytest.mark.parametrize("prominent,ingredients,expected", [
    (["Amul", "Pure Ghee"], [], "ghee"),
    (["Fortune", "Kachi Ghani", "Mustard Oil"], [], "mustard oil"),
    (["Britannia", "Butter Cookies"], [], None),                      # processed snack, not exempt
    (["Lay's", "Classic Salted"], ["Potato", "Edible Vegetable Oil", "Iodised Salt"], None),
    (["Dabur"], ["Honey"], "honey"),
])
def test_exempt_suggestion(prominent, ingredients, expected):
    assert suggest_exempt(prominent, ingredients, TH["exempt_keywords"]) == expected


# --------------------------------------------------------------- identity --

def test_brand_families():
    found = identity.find_brands("Mfd. by PepsiCo India Holdings. LAY'S Classic Salted")
    assert set(found.values()) == {"pepsico"}
    assert "britannia" in identity.find_brands("BRITANNIA Good Day").values()
    assert identity.find_brands("rich in protein, real taste") == {}      # generic words are not brands


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(identity, "off_by_barcode", lambda code: None)
    monkeypatch.setattr(identity, "off_search", lambda name: None)


def test_brand_from_other_company_is_mismatch(offline):
    out = identity.verify_packaged("BRITANNIA GOOD DAY cashew cookies nutrition", {}, "Amul Butter", None, None, TH)
    assert out["status"] == "MISMATCH"


def test_parent_company_is_not_mismatch(offline):
    out = identity.verify_packaged("Manufactured by PepsiCo India. Nutrition per 100g", {}, "Lays Magic Masala",
                                   None, None, TH)
    assert out["status"] == "VERIFIED"


def test_barcode_conflict_is_mismatch(offline):
    out = identity.verify_packaged("text", {}, None, "8901719125478", "8906002482481", TH)
    assert out["status"] == "MISMATCH"


def test_nutrient_divergence(monkeypatch):
    ref = {"code": "1", "product_name": "Snickers", "brands": "Mars",
           "nutriments": {"sugars_100g": 55, "saturated-fat_100g": 9.9, "fat_100g": 23.3, "sodium_100g": 0.244}}
    monkeypatch.setattr(identity, "off_by_barcode", lambda code: ref)
    label = {"total_sugar_g": nv("total_sugar_g", 25.5), "saturated_fat_g": nv("saturated_fat_g", 6.0),
             "total_fat_g": nv("total_fat_g", 13.0), "sodium_g": nv("sodium_g", 0.296)}
    assert identity.verify_packaged("", label, None, "1", None, TH)["status"] == "MISMATCH"
    same = {"total_sugar_g": nv("total_sugar_g", 54), "saturated_fat_g": nv("saturated_fat_g", 9.8)}
    assert identity.verify_packaged("", same, None, "1", None, TH)["status"] == "VERIFIED"


def test_no_identity_input_is_skipped(offline):
    assert identity.verify_packaged("anything", {}, "", None, None, TH)["status"] == "SKIPPED"


@pytest.mark.parametrize("name,clip_top,cnn,status", [
    ("tamatar", "tomato", "tomato", "VERIFIED"),
    ("apple", "tomato", "tomato", "MISMATCH"),
    ("Kela", "guava", "banana", "VERIFIED"),         # CLIP unsure, CNN agrees with the name
    ("", "apple", "apple", "SKIPPED"),
])
def test_verify_produce(name, clip_top, cnn, status):
    clip = {"top": clip_top, "supported": clip_top in ("tomato", "banana", "apple")}
    assert identity.verify_produce(name, clip, cnn)["status"] == status
