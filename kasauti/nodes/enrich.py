"""Classify ingredients the knowledge base doesn't know — one batched LLM call, cached per ingredient.

Results are marked origin="ai", capped at "moderate" and can never trigger a hard "avoid":
the LLM may describe an ingredient, but it doesn't get to make regulatory or allergy calls.
"""

from typing import Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from .. import cache
from ..llm import LLMError
from ..models import MatchedIngredient
from ..state import GraphState
from . import llm_from

MAX_ITEMS = 25

Category = Literal[
    "whole_food", "added_sugar", "refined_grain", "refined_carb", "oil_fat", "salt", "flavouring",
    "flavor_enhancer", "emulsifier", "thickener", "preservative", "color_natural", "color_synthetic",
    "sweetener", "industrial_protein", "acidity_regulator", "other",
]

SYSTEM = """You are a food scientist. Classify each packaged-food ingredient.
- category: the best fit from the allowed list ("whole_food" for minimally processed foods like spices, nuts, grains, dairy).
- risk for an average healthy adult eating normal amounts: "none", "low", "moderate" or "high".
  Use "moderate"/"high" ONLY when there is well-established evidence (WHO, EFSA, FDA, IARC). When unsure, use "low".
- note: one short, factual sentence. No speculation, no medical advice."""


class _Item(BaseModel):
    original: str
    name: str
    category: Category
    risk: Literal["none", "low", "moderate", "high"]
    note: str


class _Batch(BaseModel):
    items: list[_Item]


def enrich_node(state: GraphState, config: RunnableConfig) -> dict:
    llm = llm_from(config)
    unknown = state.get("unknown", [])[:MAX_ITEMS]
    if not llm or not unknown:
        return {"enriched": True}

    known: list[MatchedIngredient] = []
    todo: list[str] = []
    for u in unknown:
        hit = cache.get(cache.key("ingredient", llm.provider, llm.model, u.lower()))
        if hit:
            known.append(MatchedIngredient.model_validate(hit))
        else:
            todo.append(u)

    warnings = []
    if todo:
        try:
            batch = llm.structured(
                _Batch, SYSTEM,
                "Classify these ingredients. Return one item per input, `original` copied exactly:\n"
                + "\n".join(f"- {t}" for t in todo),
            )
            by_orig = {i.original.lower(): i for i in batch.items}
            for t in todo:
                i = by_orig.get(t.lower())
                if not i:
                    continue
                m = MatchedIngredient(
                    original=t, name=i.name, category=i.category,
                    risk="moderate" if i.risk == "high" else i.risk, note=i.note, origin="ai",
                )
                cache.put(cache.key("ingredient", llm.provider, llm.model, t.lower()), m.model_dump())
                known.append(m)
        except LLMError as e:
            warnings.append(f"Could not classify unfamiliar ingredients: {e}")

    return {"ai_matches": known, "enriched": True, "warnings": warnings}
