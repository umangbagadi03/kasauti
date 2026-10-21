from state import ACHISState

RISK_THRESHOLDS = {
    "sugar_g":   {"moderate": 10, "high": 20},
    "sodium_mg": {"moderate": 300, "high": 600},
    "fat_g":     {"moderate": 10, "high": 20},
}

ULTRA_PROCESSED_TYPES = { # set
    "preservative", "artificial_color", "flavor_enhancer",
    "emulsifier", "sweetener", "stabilizer"
}

def risk_agent(state: ACHISState) -> ACHISState:     # -> ACHISState is a hint that the function will return state
    signals = []
    nutrition = state.get("nutrition_facts", {})   # agar nutrition na mile toh crash mat maro instead {} empty provid karo
    profile = state.get("user_profile", {})

    cumulative_score = 0
    weights = {"high": 3, "moderate": 2, "low": 1}


    # NUTRITION ANALYSIS
    for nutrient, thresholds in RISK_THRESHOLDS.items():
        val = nutrition.get(nutrient, 0)

        risk_level = None
        if val >= thresholds["high"]:
            risk_level = "high"
        elif val >= thresholds["moderate"]:
            risk_level = "moderate"

        if risk_level:
            # PERSONALIZATION 
            if profile.get("condition") == "diabetic" and nutrient == "sugar_g":
                risk_level = "high"

            signals.append({
                "source": nutrient,
                "risk": risk_level,
                "value": val,
                "confidence": 0.9,
                "reason": f"{nutrient} is {val}, exceeds {risk_level} threshold"
            })

            cumulative_score += weights[risk_level]

    
    # INGREDIENT RISK
    
    for ing in state.get("decoded_ingredients", []):    # ing = ingredient
        risk = ing.get("risk", "unknown")
        confidence = ing.get("confidence", 0.5)

        if risk in ("high", "moderate"):
            signals.append({
                "source": ing["name"],
                "risk": risk,
                "confidence": confidence,
                "reason": f"{ing['name']} ({ing.get('code','')}) is a {ing['type']}"
            })

            cumulative_score += weights[risk] * confidence

    # LTRA-PROCESSED SCORE
    ingredients = state.get("decoded_ingredients", [])
    total = len(ingredients) or 1

    processed_count = sum(
        1 for i in ingredients
        if i.get("type") in ULTRA_PROCESSED_TYPES
    )

    ultra_score = round(processed_count / total, 2)

    if ultra_score > 0.6:
        signals.append({
            "source": "processing_level",
            "risk": "high",
            "confidence": 0.8,
            "reason": "Highly processed product"
        })
        cumulative_score += 3

    
    # NORMALIZE SCORE
    normalized_score = round(min(cumulative_score / 10, 1.0), 2)

    return {
        **state,
        "risk_signals": signals,
        "ultra_processed_score": ultra_score,
        "cumulative_risk_score": normalized_score,
    }
