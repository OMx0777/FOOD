from datetime import date

from foodguard import review
from foodguard.config import load_thresholds
from foodguard.fssai_engine import check_expiry, evaluate_compliance
from foodguard.label_parser import NutrientValue

TH = load_thresholds()
NO_DATE = check_expiry(None)


def nv(key, value):
    return NutrientValue(key, value, f"{key} {value}", 0)


def packaged(nutrients, category="packaged_solid", expiry=NO_DATE):
    compliance = evaluate_compliance(nutrients, category, TH)
    return review.for_packaged(nutrients, compliance, expiry, category, "Test Snack", None, ["Potato", "Oil"], "g")


def test_expiry_is_optional():
    assert check_expiry(None)["status"] == "NOT_PROVIDED"
    assert check_expiry(date(2026, 1, 1), today=date(2026, 10, 17))["status"] == "EXPIRED"
    valid = check_expiry(date(2027, 1, 1), today=date(2026, 10, 17))
    assert valid["status"] == "VALID" and valid["days_left"] == 76


def test_healthy_pack_scores_high():
    r = packaged({"added_sugar_g": nv("added_sugar_g", 0.5), "saturated_fat_g": nv("saturated_fat_g", 1.0),
                  "sodium_g": nv("sodium_g", 0.1)})
    assert r["score"] == 100 and r["rating"] == "Excellent" and r["stars"] == 5


def test_exceeded_limit_caps_score_to_match_unsafe_verdict():
    r = packaged({"added_sugar_g": nv("added_sugar_g", 1.0), "saturated_fat_g": nv("saturated_fat_g", 4.5),
                  "sodium_g": nv("sodium_g", 0.2)})
    assert r["score"] <= review.EXCEEDED_CAP
    assert r["rating"] in ("Poor", "Avoid")


def test_expired_pack_scores_zero():
    expired = check_expiry(date(2025, 1, 1), today=date(2026, 10, 17))
    r = packaged({"added_sugar_g": nv("added_sugar_g", 0.5)}, expiry=expired)
    assert r["score"] == 0 and r["rating"] == "Avoid"


def test_exempt_is_not_scored():
    r = packaged({"saturated_fat_g": nv("saturated_fat_g", 62.0)}, category="packaged_exempt")
    assert r["score"] is None and r["rating"] == "Exempt"


def test_packaged_info_and_daily_share():
    r = packaged({"energy_kcal": nv("energy_kcal", 500), "sodium_g": nv("sodium_g", 1.0)})
    info = r["info"]
    assert info["title"] == "Test Snack"
    assert {"label": "Sodium", "value": "1000 mg"} in info["facts"]
    shares = {d["name"]: d["percent"] for d in info["daily_share"]}
    assert shares == {"Energy": 25, "Sodium": 50}


def test_produce_review_uses_freshness_and_knowledge_base():
    result = {"produce_type": "tomato", "fresh_probability": 0.93, "surface_analysis": {"dark_patch_percent": 2.0}}
    r = review.for_produce(result)
    assert r["score"] == 93 and r["rating"] == "Excellent"
    assert r["info"]["title"] == "Tomato"
    assert any(f["label"] == "Vitamin C" for f in r["info"]["facts"])
    assert r["info"]["storage"]
