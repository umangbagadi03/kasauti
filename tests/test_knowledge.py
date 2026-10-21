from kasauti.knowledge import detect_allergens, match_ingredients, split_ingredients
from kasauti.knowledge.rules import claim_signals, nutrient_signals
from kasauti.models import Nutrition


def names(items):
    return {m.name for m in match_ingredients(items).matched}


def test_split_respects_brackets():
    assert split_ingredients("Ingredients: Sugar, Emulsifiers (322, 471), Oil (Palmolein); Salt.") == [
        "Sugar", "Emulsifiers (322, 471)", "Oil (Palmolein)", "Salt"
    ]


def test_codes_in_every_label_style():
    found = names(["Emulsifiers (322, 471)", "Colour (INS 150d)", "E-211", "Antioxidant (319)", "raising agent 503(ii)"])
    assert {"Lecithin", "Mono- and Diglycerides of Fatty Acids", "Sulphite Ammonia Caramel",
            "Sodium Benzoate", "TBHQ", "Ammonium Carbonates"} <= found


def test_percentages_and_quantities_are_not_codes():
    out = match_ingredients(["Cocoa Solids (100%)", "Sugar 500mg"])
    assert all(m.code is None for m in out.matched)


def test_longest_phrase_wins():
    assert names(["Refined Wheat Flour"]) == {"Refined Wheat Flour (Maida)"}
    assert names(["Whole Wheat Flour"]) == {"Whole Wheat Flour"}
    assert names(["Invert Sugar Syrup"]) == {"Invert Sugar"}


def test_group_label_with_children_is_not_double_counted_or_unknown():
    out = match_ingredients(["Emulsifiers (322)", "Cream filling (Sugar, Cocoa)"])
    assert "Emulsifier (unspecified)" not in {m.name for m in out.matched}
    assert out.unknown == []


def test_unknowns_reported():
    assert match_ingredients(["Kokum Extract"]).unknown == ["Kokum Extract"]


def test_allergen_false_friends():
    a = detect_allergens(["Cocoa Butter", "Coconut Milk", "Maltodextrin", "Peanut Butter"], [])
    assert "milk" not in a and "gluten" not in a and a["peanut"] == "contains"


def test_may_contain_is_trace():
    a = detect_allergens(["Milk Solids"], ["May contain traces of nuts"])
    assert a == {"milk": "contains", "tree_nuts": "may_contain"}


def test_diabetes_tightens_sugar_threshold():
    n = Nutrition(sugar_g=8)
    assert [s.level for s in nutrient_signals(n, False, [])] == ["moderate"]
    assert [s.level for s in nutrient_signals(n, False, ["diabetes"])] == ["moderate"]
    assert [s.level for s in nutrient_signals(Nutrition(sugar_g=12), False, ["diabetes"])] == ["high"]


def test_liquid_thresholds_are_halved():
    assert [s.level for s in nutrient_signals(Nutrition(sugar_g=12), True, [])] == ["high"]
    assert [s.level for s in nutrient_signals(Nutrition(sugar_g=12), False, [])] == ["moderate"]


def test_claims():
    n = Nutrition(protein_g=6, energy_kcal=480)
    out = claim_signals(["No added sugar", "Multigrain", "High protein"], ["added_sugar", "refined_grain"],
                        "refined_grain", n, False)
    titles = " ".join(s.title for s in out)
    assert "No added sugar" in titles and "Multigrain" in titles and "High protein" in titles


def test_no_added_preservatives_claim():
    out = claim_signals(["No added preservatives"], ["preservative"], None, Nutrition(), False)
    assert [s.level for s in out] == ["high"]


def test_mixed_contains_and_may_contain_statement():
    a = detect_allergens([], ["Contains: Wheat, Milk, Soy. May contain traces of nuts."])
    assert a == {"gluten": "contains", "milk": "contains", "soy": "contains", "tree_nuts": "may_contain"}
