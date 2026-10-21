import json
import re
from typing import List, Dict
from state import ACHISState

with open("data/additives_db.json") as f:
    ADDITIVES_DB = json.load(f)

# 🔥 EXPANDED ALLERGEN MAP (NOT JUST WORD MATCH)
ALLERGEN_MAP = {
    "milk": ["milk", "casein", "whey", "lactose"],
    "egg": ["egg", "albumin"],
    "soy": ["soy", "soya", "lecithin"],
    "wheat": ["wheat", "gluten"],
    "peanut": ["peanut", "groundnut"],
    "tree_nuts": ["almond", "cashew", "walnut"],
    "fish": ["fish"],
    "sesame": ["sesame"],
}

# main agent
def ingredient_agent(state: ACHISState) -> ACHISState:
    decoded = []
    allergens_found = set()
    unknown_ingredients = []

    for ingredient in state.get("raw_ingredients", []):   # says: "Hey, look for the ingredients. If they aren't there, just give me an empty list [] so I don't break."
        ing_lower = ingredient.lower().strip()    # LOWERING ingredient

        matches = _match_additives(ing_lower) # db_matching

        if matches:
            for match in matches:
                decoded.append(match)
        else:
            unknown_ingredients.append(ingredient)
            decoded.append({
                "original": ingredient,
                "name": ingredient,
                "type": "unknown",
                "risk": "unknown",
                "code": None,
                "confidence": 0.4
            })
        # Allergen detection (context aware)
        detected = _detect_allergens(ing_lower)
        allergens_found.update(detected)
    return {
        **state,
        "decoded_ingredients": decoded,
        "allergen_flags": list(allergens_found),
        "unknown_ingredients": unknown_ingredients,
    }


# ADDITIVE MATCHING
def _match_additives(text: str) -> List[Dict]:
    results = []

    for code, info in ADDITIVES_DB.items():
        code_match = re.search(rf"\b{code.lower()}\b", text)
        name_match = info["name"].lower() in text

        if code_match or name_match:
            results.append({
                "original": text,
                "name": info["name"],
                "type": info["type"],
                "risk": info["risk"],
                "code": code,
                "confidence": 0.95 if code_match else 0.75
            })
    return results


# ALLERGEN DETECTION
def _detect_allergens(text: str) -> List[str]:
    detected = []

    for allergen, keywords in ALLERGEN_MAP.items():  # The .items() method is used with dictionaries in Python to iterate over both the keys and values of the dictionary as key-value pairs.
        for keyword in keywords:
            if keyword in text:
                if "may contains" in text:
                    detected.append(f"{allergen}_trace")
                else:
                    detected.append(allergen)

    return detected




