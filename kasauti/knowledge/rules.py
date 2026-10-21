"""Deterministic health rules: nutrient thresholds, condition overrides, claim checks.

Nutrient thresholds follow the UK FSA front-of-pack "traffic light" criteria
(per 100 g; halved for drinks per 100 ml). Condition overrides are stricter
cut-offs commonly used in clinical diet advice — tune them with a dietitian.
"""

import re

from ..models import Nutrition, Signal

# nutrient -> (moderate_above, high_above) per 100 g of food
THRESHOLDS: dict[str, tuple[float, float]] = {
    "sugar_g": (5.0, 22.5),
    "fat_g": (3.0, 17.5),
    "saturated_fat_g": (1.5, 5.0),
    "sodium_mg": (120.0, 600.0),  # = salt 0.3 g / 1.5 g
}
LIQUID_FACTOR = 0.5

CONDITION_THRESHOLDS: dict[str, dict[str, tuple[float, float]]] = {
    "diabetes": {"sugar_g": (2.5, 10.0)},
    "weight_loss": {"sugar_g": (5.0, 15.0), "fat_g": (3.0, 15.0)},
    "hypertension": {"sodium_mg": (120.0, 400.0)},
    "kidney_disease": {"sodium_mg": (120.0, 400.0)},
}

LABELS = {"sugar_g": "Sugar", "fat_g": "Fat", "saturated_fat_g": "Saturated fat", "sodium_mg": "Sodium"}
UNITS = {"sugar_g": "g", "fat_g": "g", "saturated_fat_g": "g", "sodium_mg": "mg"}

TRANS_FAT_HIGH_G = 0.2   # "trans-fat free" claims require < 0.2 g / 100 g
HIGH_FIBRE_G = 6.0       # EU "high fibre"
ENERGY_DENSE_KCAL = 400.0


def nutrient_signals(n: Nutrition, is_liquid: bool, conditions: list[str]) -> list[Signal]:
    signals: list[Signal] = []
    basis = "100 ml" if is_liquid else "100 g"
    factor = LIQUID_FACTOR if is_liquid else 1.0

    for key, (mod, high) in THRESHOLDS.items():
        value = getattr(n, key)
        if value is None:
            continue
        stricter = [(c, CONDITION_THRESHOLDS[c][key]) for c in conditions if key in CONDITION_THRESHOLDS.get(c, {})]
        reason = ""
        if stricter:
            cond, (mod, high) = min(stricter, key=lambda t: t[1][1])
            reason = f" (stricter limit for {cond.replace('_', ' ')})"
        mod, high = mod * factor, high * factor

        level = "high" if value > high else "moderate" if value > mod else None
        if level:
            unit = UNITS[key]
            signals.append(Signal(
                kind="nutrient", level=level,
                title=f"{'High' if level == 'high' else 'Medium'} {LABELS[key].lower()}",
                detail=f"{LABELS[key]} is {value:g}{unit} per {basis}; above {high if level == 'high' else mod:g}{unit} counts as {level}{reason}.",
                evidence=key,
            ))

    if n.trans_fat_g is not None and n.trans_fat_g >= TRANS_FAT_HIGH_G:
        signals.append(Signal(
            kind="nutrient", level="high", title="Contains trans fat",
            detail=f"Trans fat is {n.trans_fat_g:g}g per {basis}. There is no safe level of industrial trans fat (WHO).",
            evidence="trans_fat_g",
        ))

    if "weight_loss" in conditions and n.energy_kcal and n.energy_kcal > ENERGY_DENSE_KCAL * factor:
        signals.append(Signal(
            kind="condition", level="moderate", title="Energy-dense",
            detail=f"{n.energy_kcal:g} kcal per {basis} — easy to overeat when trying to lose weight.",
            evidence="energy_kcal",
        ))
    return signals


def positive_points(n: Nutrition) -> tuple[float, list[str]]:
    """Small credit for genuinely good things; returns (points, reasons)."""
    if n.fiber_g is not None and n.fiber_g >= HIGH_FIBRE_G:
        return 1.0, [f"High in fibre ({n.fiber_g:g}g per 100g)"]
    return 0.0, []


# ---------------------------------------------------------------- claim checks

def _has(claims: list[str], *phrases: str) -> str | None:
    for c in claims:
        for p in phrases:
            if re.search(p, c, re.I):
                return c
    return None


def claim_signals(claims: list[str], categories: list[str], first_category: str | None, n: Nutrition, is_liquid: bool) -> list[Signal]:
    """Compare front-of-pack claims with what the ingredients/nutrition actually say."""
    out: list[Signal] = []
    cats = set(categories)

    def flag(claim: str, level: str, detail: str):
        out.append(Signal(kind="claim", level=level, title=f'Misleading claim: "{claim}"', detail=detail, evidence=claim))

    if c := _has(claims, r"no added sugar", r"no sugar added", r"sugar[- ]free", r"zero sugar"):
        if "added_sugar" in cats:
            flag(c, "high", "The ingredient list contains added sugars or syrups (e.g. jaggery, honey, glucose, juice concentrate).")
        elif re.search(r"sugar[- ]free|zero sugar", c, re.I) and n.sugar_g is not None and n.sugar_g > 0.5:
            flag(c, "high", f"Label shows {n.sugar_g:g}g sugar per 100{'ml' if is_liquid else 'g'}; 'sugar free' means under 0.5g.")

    if c := _has(claims, r"multi ?grain", r"whole ?wheat", r"whole ?grain", r"\batta\b", r"millet"):
        if first_category == "refined_grain":
            flag(c, "moderate", "The first (largest) ingredient is refined flour, not whole grain.")

    if c := _has(claims, r"\bnatural\b", r"no artificial"):
        artificial = cats & {"color_synthetic", "flavouring_artificial", "sweetener", "antioxidant_synthetic"}
        if artificial:
            flag(c, "moderate", f"Contains {', '.join(sorted(a.replace('_', ' ') for a in artificial))}.")

    if c := _has(claims, r"no (?:added )?preservatives?", r"preservative[- ]free"):
        if "preservative" in cats:
            flag(c, "high", "A preservative appears in the ingredient list.")

    if c := _has(claims, r"no artificial colou?rs?", r"no synthetic colou?rs?"):
        if "color_synthetic" in cats:
            flag(c, "high", "A synthetic colour appears in the ingredient list.")

    if c := _has(claims, r"trans[- ]fat[- ]free", r"zero trans", r"0\s?g? trans", r"no trans"):
        if "trans_fat" in cats:
            flag(c, "high", "Contains hydrogenated fat, the main source of trans fat.")

    if c := _has(claims, r"high protein", r"protein rich", r"rich in protein", r"source of protein"):
        if n.protein_g is not None:
            share = (n.protein_g * 4 / n.energy_kcal) if n.energy_kcal else None
            weak = share < 0.20 if share is not None else n.protein_g < 10
            if weak:
                pct = f" ({share:.0%} of calories)" if share is not None else ""
                flag(c, "moderate", f"Only {n.protein_g:g}g protein per 100g{pct}; 'high protein' usually means at least 20% of calories.")

    if c := _has(claims, r"high fib(?:re|er)"):
        if n.fiber_g is not None and n.fiber_g < HIGH_FIBRE_G:
            flag(c, "moderate", f"Only {n.fiber_g:g}g fibre per 100g; 'high fibre' means at least 6g.")

    if c := _has(claims, r"low fat"):
        limit = 1.5 if is_liquid else 3.0
        if n.fat_g is not None and n.fat_g > limit:
            flag(c, "moderate", f"Contains {n.fat_g:g}g fat per 100{'ml' if is_liquid else 'g'}; 'low fat' means at most {limit:g}g.")

    if c := _has(claims, r"baked,? not fried", r"not fried"):
        if n.fat_g is not None and n.fat_g > 17.5:
            flag(c, "low", f"Still high in fat ({n.fat_g:g}g per 100g) despite being baked.")

    return out
