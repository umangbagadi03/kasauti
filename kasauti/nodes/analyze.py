"""Deterministic analysis: every signal here is traceable to the label and the curated knowledge base."""

from ..knowledge import UPF_CATEGORIES, detect_allergens, match_ingredients
from ..knowledge.rules import claim_signals, nutrient_signals, positive_points
from ..models import Signal, UserProfile
from ..state import GraphState

RANK = {"none": 0, "low": 1, "moderate": 2, "high": 3, "avoid": 4}
STARCHY = {"added_sugar", "refined_grain", "refined_carb"}

ALLERGY_ALIASES = {
    "dairy": "milk", "lactose": "milk", "eggs": "egg", "soya": "soy", "wheat": "gluten",
    "peanuts": "peanut", "groundnut": "peanut", "nuts": "tree_nuts", "tree nuts": "tree_nuts",
    "tree_nut": "tree_nuts", "shellfish": "crustacean", "sulfites": "sulphites", "sulphite": "sulphites",
}


def normalize_allergies(allergies: list[str]) -> set[str]:
    out = set()
    for a in allergies:
        a = a.strip().lower()
        out.add(ALLERGY_ALIASES.get(a, a))
    return out


def analyze_node(state: GraphState) -> dict:
    label = state.get("label")
    if not label or (not label.ingredients and not label.nutrition.model_dump(exclude_none=True)):
        return {"matched": [], "unknown": [], "signals": [], "positives": [], "allergens": {}, "ultra_processed": False}

    profile = state.get("profile") or UserProfile()
    conds = list(profile.conditions)
    outcome = match_ingredients(label.ingredients)

    ai_by_name = {m.original.lower(): m for m in state.get("ai_matches", [])}
    ai_used = [ai_by_name[u.lower()] for u in outcome.unknown if u.lower() in ai_by_name]
    unknown = [u for u in outcome.unknown if u.lower() not in ai_by_name]

    signals: list[Signal] = []
    n = label.nutrition

    # ---- ingredients from the curated knowledge base
    for key, e in outcome.entries.items():
        pos = outcome.positions[key]
        level, kind, why = e.risk, "ingredient", ""
        for c in conds:
            cl = e.conditions.get(c)
            if cl and RANK[cl] > RANK[level]:
                level, kind, why = cl, "condition", c.replace("_", " ")
        detail = e.note
        if kind == "condition" and e.category == "added_sugar" and n.sugar_g is not None:
            level = "moderate"  # the sugar number in the nutrition table carries the weight
        elif kind == "condition" and e.category in STARCHY and pos < 3 and level == "moderate":
            level, detail = "high", f"{e.note} It is one of the main ingredients."
        if RANK[level] >= RANK["moderate"]:
            title = f"{e.name}{f' ({e.code})' if e.code else ''}"
            signals.append(Signal(
                kind=kind, level=level, title=title + (f" — matters for {why}" if why else ""),
                detail=detail, evidence=", ".join(e.sources),
            ))
        if e.not_permitted_in:
            if profile.region in e.not_permitted_in:
                signals.append(Signal(
                    kind="regulatory", level="avoid", title=f"{e.name} is not permitted in {profile.region}",
                    detail=e.note, evidence=", ".join(e.sources),
                ))
            else:
                signals.append(Signal(
                    kind="regulatory", level="low", title=f"{e.name} is not permitted in {', '.join(e.not_permitted_in)}",
                    detail=e.note, evidence=", ".join(e.sources),
                ))

    # ---- ingredients classified by the LLM: shown, but capped and clearly marked
    for m in ai_used:
        if RANK[m.risk] >= RANK["moderate"]:
            signals.append(Signal(
                kind="ingredient", level="moderate", title=f"{m.name} (AI-classified, unverified)", detail=m.note,
            ))

    # ---- sugar as a main ingredient when the nutrition table is missing
    if n.sugar_g is None and any(
        e.category == "added_sugar" and outcome.positions[k] < 3 for k, e in outcome.entries.items()
    ):
        signals.append(Signal(
            kind="ingredient", level="high" if {"diabetes", "weight_loss"} & set(conds) else "moderate",
            title="Sugar is a main ingredient", detail="Added sugar is among the first three ingredients (the label gives no sugar amount).",
        ))

    # ---- allergens: only a problem if the user is allergic
    allergens = detect_allergens(label.ingredients, label.allergen_statements)
    for e in outcome.entries.values():
        if e.allergen:
            allergens[e.allergen] = "contains"
    mine = normalize_allergies(profile.allergies)
    if "celiac" in conds:
        mine.add("gluten")
    for a in sorted(mine & allergens.keys()):
        contains = allergens[a] == "contains"
        signals.append(Signal(
            kind="allergen", level="avoid" if contains else "high",
            title=f"{'Contains' if contains else 'May contain'} {a.replace('_', ' ')}",
            detail="You listed this as an allergy/intolerance." if a != "gluten" or "celiac" not in conds
            else "Gluten must be avoided with celiac disease.",
        ))

    # ---- nutrition
    signals += nutrient_signals(n, label.is_liquid, conds)
    _, positives = positive_points(n)

    # ---- processing level (NOVA-style markers)
    markers = [e.name for e in outcome.entries.values() if e.upf]
    markers += [m.name for m in ai_used if m.category in UPF_CATEGORIES]
    if markers:
        level = "high" if len(markers) >= 4 else "moderate" if len(markers) >= 2 else "low"
        signals.append(Signal(
            kind="processing", level=level, title="Ultra-processed",
            detail=f"{len(markers)} industrial ingredient(s)/cosmetic additive(s): {', '.join(markers[:8])}.",
        ))

    # ---- marketing claims vs reality
    categories = [e.category for e in outcome.entries.values()] + [m.category for m in ai_used]
    first = next((e.category for k, e in outcome.entries.items() if outcome.positions[k] == 0), None)
    signals += claim_signals(label.claims, categories, first, n, label.is_liquid)

    return {
        "matched": outcome.matched + ai_used,
        "unknown": unknown,
        "signals": signals,
        "positives": positives,
        "allergens": allergens,
        "ultra_processed": bool(markers),
    }
