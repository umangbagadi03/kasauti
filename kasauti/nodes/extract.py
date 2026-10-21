"""Read the label: one multimodal LLM call (image or text), or a regex parser offline."""

import re

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, field_validator

from .. import cache
from ..knowledge import split_ingredients
from ..llm import LLMClient, LLMError
from ..models import LabelData, Nutrition
from ..state import GraphState
from . import llm_from

SYSTEM = """You transcribe packaged-food labels into structured data. You are precise and never guess.
Rules:
- ingredients: the ingredient list in label order, one entry per top-level ingredient. Keep bracketed
  sub-ingredients and E/INS numbers exactly, e.g. "Emulsifiers (322, 471)". Translate to English if needed.
- nutrition: values PER 100 g (or per 100 ml for drinks). If the label only gives per-serving values and
  the serving size, convert to per 100. If salt is given instead of sodium, sodium_mg = salt_g * 400.
  Use null for anything not printed. Never invent values.
- claims: marketing claims printed on the pack ("No added sugar", "Multigrain", "High protein"...).
- allergen_statements: allergen lines verbatim ("Contains: milk, wheat", "May contain traces of nuts").
- legibility: 1.0 if everything was clearly readable, lower if blurred/cropped/partially visible.
- is_food_label: false if this is not a food/drink label at all."""


PROMPT_VERSION = "2"  # bump when SYSTEM/Extraction change, to invalidate cached extractions


class Extraction(BaseModel):
    is_food_label: bool = True
    product_name: str | None = None
    brand: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    nutrition: Nutrition = Field(default_factory=Nutrition)
    is_liquid: bool = False
    claims: list[str] = Field(default_factory=list)
    allergen_statements: list[str] = Field(default_factory=list)
    legibility: float = Field(default=1.0, ge=0, le=1)

    @field_validator("product_name", "brand")
    @classmethod
    def _no_placeholders(cls, v: str | None) -> str | None:
        """Models sometimes write 'Unknown' / 'No brand information available' instead of null."""
        if v and re.search(r"^\W*(unknown|n/?a|none|null|not (?:available|visible|provided)|no \w+ (?:information|name)|<.*>)", v.strip(), re.I):
            return None
        return v or None


def extract_node(state: GraphState, config: RunnableConfig) -> dict:
    image, text = state.get("image"), state.get("text")
    if not image and not text:
        return {"warnings": ["Nothing to analyse — provide a label photo, label text or a barcode."]}

    llm = llm_from(config)
    if llm is None:
        if image:
            return {"warnings": ["Reading photos needs an LLM provider (offline mode supports text only)."]}
        return {"label": parse_label_text(text)}

    source = "image" if image else "text"
    try:
        ex = _extract_cached(llm, image.data if image else text.encode(), image, text)
    except LLMError as e:
        if text:  # degrade gracefully for text input
            return {"label": parse_label_text(text), "warnings": [f"LLM extraction failed, used basic parser: {e}"]}
        return {"warnings": [f"Could not read the label: {e}"]}

    if not ex.is_food_label:
        return {"warnings": ["That doesn't look like a food label."]}
    label = LabelData(**ex.model_dump(exclude={"is_food_label"}), source=source)
    if text and not image:
        # Small models sometimes drop a value; the regex parser is a cheap cross-check for typed text.
        parsed = parse_label_text(text).nutrition
        gaps = {k: v for k, v in parsed.model_dump().items() if v is not None and getattr(label.nutrition, k) is None}
        if gaps:
            label.nutrition = label.nutrition.model_copy(update=gaps)
    warnings = ["The photo was hard to read — results may be incomplete."] if label.legibility < 0.6 else []
    return {"label": label, "warnings": warnings}


def _extract_cached(llm: LLMClient, payload: bytes, image, text) -> Extraction:
    k = cache.key("extract", PROMPT_VERSION, llm.provider, llm.model, payload)
    if (hit := cache.get(k)) is not None:
        return Extraction.model_validate(hit)
    prompt = "Transcribe this food label." if image else f"Transcribe this food label text:\n\n{text}"
    ex = llm.structured(Extraction, SYSTEM, prompt, image=image)
    if ex.ingredients or ex.nutrition.model_dump(exclude_none=True):  # never cache an empty read
        cache.put(k, ex.model_dump())
    return ex


# ---------------------------------------------------------------- offline parser

_NUTRIENTS = {
    "energy_kcal": r"(?:energy|calories)[^\d]{0,20}([\d.]+)\s*k?cal",
    "sugar_g": r"(?<!added )(?:total )?sugars?[^\d]{0,15}([\d.]+)\s*g",
    "added_sugar_g": r"added sugars?[^\d]{0,15}([\d.]+)\s*g",
    "saturated_fat_g": r"saturated(?: fat| fatty acids)?[^\d]{0,15}([\d.]+)\s*g",
    "trans_fat_g": r"trans(?: fat| fatty acids)?[^\d]{0,15}([\d.]+)\s*g",
    "fat_g": r"(?<!saturated )(?<!trans )(?:total )?fat[^\d]{0,15}([\d.]+)\s*g",
    "sodium_mg": r"sodium[^\d]{0,15}([\d.]+)\s*mg",
    "protein_g": r"protein[^\d]{0,15}([\d.]+)\s*g",
    "fiber_g": r"(?:dietary )?fib(?:re|er)[^\d]{0,15}([\d.]+)\s*g",
    "carbohydrate_g": r"carbohydrates?[^\d]{0,15}([\d.]+)\s*g",
}


def parse_label_text(text: str) -> LabelData:
    """Best-effort parser for typed/pasted label text. Assumes the first number is per 100 g."""
    low = text.lower()
    m = re.search(r"ingredients?\s*[:\-]\s*(.+?)(?=\n\s*\n|nutrition|nutritional|allergen|contains:|may contain|$)", low, re.S)
    ingredients = split_ingredients(m.group(1).replace("\n", " ")) if m else []

    nutri = low[low.find("nutrition"):] if "nutrition" in low else low
    values = {}
    for key, pattern in _NUTRIENTS.items():
        if mm := re.search(pattern, nutri):
            try:
                values[key] = float(mm.group(1).rstrip("."))
            except ValueError:
                pass
    if "sodium_mg" not in values and (mm := re.search(r"\bsalt[^\d]{0,15}([\d.]+)\s*g", nutri)):
        values["sodium_mg"] = round(float(mm.group(1)) * 400, 1)

    statements = [s.strip() for s in re.findall(r"((?:allergen[^:]*:|contains:|may contain)[^\n.]*)", low)]
    return LabelData(
        ingredients=ingredients, nutrition=Nutrition(**values), allergen_statements=statements,
        is_liquid=bool(re.search(r"per\s*100\s*ml", low)), legibility=0.8 if ingredients else 0.3, source="text",
    )
