import pytest

from foodguard.config import load_thresholds
from foodguard.label_parser import match_nutrient, parse_ingredients, parse_nutrients

RULES = load_thresholds()["nutrient_rules"]


def values(lines):
    return {k: round(v.value, 4) for k, v in parse_nutrients(lines, RULES)["nutrients"].items()}


@pytest.mark.parametrize("label,expected", [
    ("Saturated Fat", "saturated_fat_g"),
    ("Saturated Fal", "saturated_fat_g"),        # OCR slip
    ("Sodiurn", "sodium_g"),
    ("Added Sugars", "added_sugar_g"),
    ("Total Sugars", "total_sugar_g"),
    ("Total Fat (g)", "total_fat_g"),
    ("Trans Fat", "trans_fat_g"),
    ("Ingredients: wheat flour, sugar, edible vegetable oil", None),
])
def test_match_nutrient(label, expected):
    assert match_nutrient(label)[0] == expected


def test_typical_indian_table_per100_first():
    lines = [
        "Nutritional Information",
        "Per 100 g  Per serve (30 g)  %RDA per serve",
        "Energy 544 kcal 163 kcal 8%",
        "Total Fat 33.9 g 10.2 g 15%",
        "Saturated Fat 15.2 g 4.6 g 21%",
        "Trans Fat 0.1 g 0.03 g",
        "Total Sugars 2.1 g 0.6 g",
        "Added Sugars 1.0 g 0.3 g 1%",
        "Sodium 720 mg 216 mg 11%",
    ]
    v = values(lines)
    assert v["saturated_fat_g"] == 15.2
    assert v["added_sugar_g"] == 1.0
    assert v["sodium_g"] == 0.72
    assert v["energy_kcal"] == 544


def test_per_serve_column_first():
    lines = ["Per serve (25 g)  Per 100 g", "Sodium 100 mg 400 mg", "Added Sugars 2 g 8 g"]
    v = values(lines)
    assert v["sodium_g"] == 0.4
    assert v["added_sugar_g"] == 8


def test_per_serving_only_is_normalised():
    lines = ["Serving size: 40 g", "Amount per serving", "Saturated Fat 2 g", "Sodium 200 mg"]
    v = values(lines)
    assert v["saturated_fat_g"] == 5.0
    assert v["sodium_g"] == 0.5


def test_salt_converted_to_sodium_and_ocr_digit_fix():
    v = values(["Per 100 g", "Salt 1.O g", "Saturated fat O.5g"])
    assert v["sodium_g"] == 0.4
    assert v["saturated_fat_g"] == 0.5


def test_liquid_basis_detected():
    out = parse_nutrients(["Nutrition Information per 100 ml", "Added Sugar 10.6 g"], RULES)
    assert out["layout"].basis == "ml"


def test_impossible_value_dropped():
    out = parse_nutrients(["Per 100 g", "Saturated Fat 152 g"], RULES)
    assert "saturated_fat_g" not in out["nutrients"]
    assert out["warnings"]


def test_ingredients_top_level_split():
    lines = ["INGREDIENTS: Potato, Edible Vegetable Oil (Palmolein, Rice Bran Oil), Iodised Salt.",
             "Nutritional information"]
    assert parse_ingredients(lines) == ["Potato", "Edible Vegetable Oil (Palmolein, Rice Bran Oil)", "Iodised Salt"]


def test_unit_column_read_as_nine_is_dropped():
    v = values(["Nutrition Information Per 100g", "Saturated Fat 9 14.0", "Total Fat (9) 35.5"])
    assert v["saturated_fat_g"] == 14.0
    assert v["total_fat_g"] == 35.5


def test_lost_decimal_restored_from_per_serve_column():
    lines = ["Per 100 g  Per serve (30 g)", "Total Sugars 33 1.0 g", "Sodium 520 mg 156 mg"]
    v = values(lines)
    assert v["total_sugar_g"] == 3.3
    assert v["sodium_g"] == 0.52


def test_trailing_e_read_as_gram():
    assert values(["Per 100 g", "Salt 0.78e"])["sodium_g"] == round(0.78 / 2.5, 4)


def test_leading_zero_and_g_read_as_9():
    v = values(["Per 100 g", "Total Sugars 040", "Total Fat 33.29", "Sodium 83m9"])
    assert v["total_sugar_g"] == 0.4
    assert v["total_fat_g"] == 33.2
    assert v["sodium_g"] == 0.083


def test_per_serving_only_table_with_two_columns():
    lines = ["Per 30 g serving   30 g serving with Milk", "Saturated Fatty Acids 0.3 g 1.2 g", "Sodium 0.15 g 0.21 g"]
    v = values(lines)
    assert v["saturated_fat_g"] == 1.0
    assert v["sodium_g"] == 0.5


def test_missed_per100_header_is_not_scaled_into_impossible_values():
    # OCR lost the "Per 100 g" header but read the "Per serve" one; 454 kcal x 100/15 would be impossible.
    lines = ["Per serve (15 g)", "ENERGY (kcal) 454.0 68.1", "ADDED SUGARS (0) 25.0 7.5",
             "SATURATED FAT (g) 6.0 0.9", "SODIUM (mg) 296 44"]
    v = values(lines)
    assert v["energy_kcal"] == 454.0
    assert v["added_sugar_g"] == 25.0
    assert v["saturated_fat_g"] == 6.0
    assert v["sodium_g"] == 0.296


def test_column_ratio_proves_first_column_is_per100():
    lines = ["Per serve", "If a serve is 70 g", "Saturated Fat 8.2 5.7", "Total Sugars 1.8 1.3"]
    v = values(lines)
    assert v["saturated_fat_g"] == 8.2
    assert v["total_sugar_g"] == 1.8
