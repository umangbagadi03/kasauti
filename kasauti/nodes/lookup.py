"""Barcode → Open Food Facts. Free, no key; cached so each product is fetched once a week."""

import re

import httpx

from .. import cache
from ..models import LabelData, Nutrition
from ..state import GraphState

OFF_URL = "https://world.openfoodfacts.org/api/v2/product/{code}.json"
FIELDS = "product_name,brands,ingredients_text_en,ingredients_text,nutriments,allergens_tags,traces_tags,labels_tags"
HEADERS = {"User-Agent": "Kasauti/0.2 (food label analyzer)"}
WEEK = 7 * 24 * 3600


def lookup_node(state: GraphState) -> dict:
    code = re.sub(r"\D", "", state.get("barcode") or "")
    if not code:
        return {"warnings": ["Barcode is not numeric — ignored."]}

    k = cache.key("off", code)
    data = cache.get(k, max_age_s=WEEK)
    if data is None:
        try:
            r = httpx.get(OFF_URL.format(code=code), params={"fields": FIELDS}, headers=HEADERS, timeout=10)
            data = r.json() if r.status_code in (200, 404) else None
        except (httpx.HTTPError, ValueError):
            data = None
        if data is None:
            return {"warnings": ["Open Food Facts is unreachable — falling back to the label."]}
        cache.put(k, data)

    product = data.get("product") if data.get("status") == 1 else None
    if not product:
        return {"warnings": [f"Barcode {code} not found in Open Food Facts."]}

    label = off_to_label(product)
    if not label.ingredients:
        return {"warnings": [f"Open Food Facts has no ingredient list for {code}."]}
    return {"label": label}


def off_to_label(p: dict) -> LabelData:
    from ..knowledge import split_ingredients

    n = p.get("nutriments", {})

    def g(name: str, scale: float = 1.0):
        v = n.get(f"{name}_100g")
        return round(float(v) * scale, 3) if isinstance(v, (int, float)) else None

    text = p.get("ingredients_text_en") or p.get("ingredients_text") or ""
    tags = lambda key: [t.split(":", 1)[-1].replace("-", " ") for t in p.get(key, [])]  # noqa: E731
    return LabelData(
        product_name=p.get("product_name") or None,
        brand=(p.get("brands") or "").split(",")[0].strip() or None,
        ingredients=split_ingredients(text) if text else [],
        nutrition=Nutrition(
            energy_kcal=g("energy-kcal"), sugar_g=g("sugars"), fat_g=g("fat"),
            saturated_fat_g=g("saturated-fat"), trans_fat_g=g("trans-fat"),
            sodium_mg=g("sodium", 1000),  # OFF stores sodium in grams
            protein_g=g("proteins"), fiber_g=g("fiber"), carbohydrate_g=g("carbohydrates"),
        ),
        claims=tags("labels_tags"),
        allergen_statements=[f"contains {a}" for a in tags("allergens_tags")]
        + [f"may contain {t}" for t in tags("traces_tags")],
        legibility=0.9,  # crowd-sourced data: good, not perfect
        source="barcode",
    )
