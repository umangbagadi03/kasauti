"""Typed data shared across the pipeline and the API."""

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["Safe", "Moderate", "Risky", "Unknown"]
Level = Literal["low", "moderate", "high", "avoid"]

Condition = Literal[
    "diabetes", "hypertension", "kidney_disease", "pku", "celiac", "pregnancy", "gout", "weight_loss"
]


class UserProfile(BaseModel):
    conditions: list[Condition] = Field(default_factory=list)
    # Allergens the user actually reacts to, e.g. ["milk", "peanut"]. Keys of ALLERGENS.
    allergies: list[str] = Field(default_factory=list)
    # Where the user lives — decides which "not permitted" rules are hard stops.
    region: str = "IN"
    # Language for the explanation, e.g. "English", "Hindi".
    language: str = "English"


class Nutrition(BaseModel):
    """Values per 100 g (solids) or per 100 ml (liquids). None = not on label."""

    energy_kcal: float | None = None
    sugar_g: float | None = None
    added_sugar_g: float | None = None
    fat_g: float | None = None
    saturated_fat_g: float | None = None
    trans_fat_g: float | None = None
    sodium_mg: float | None = None
    protein_g: float | None = None
    fiber_g: float | None = None
    carbohydrate_g: float | None = None


class LabelData(BaseModel):
    """What the extractor read off the package (or Open Food Facts returned)."""

    product_name: str | None = None
    brand: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    nutrition: Nutrition = Field(default_factory=Nutrition)
    is_liquid: bool = False
    # Front-of-pack marketing claims, verbatim: "no added sugar", "multigrain", ...
    claims: list[str] = Field(default_factory=list)
    # Allergen statements printed on the pack ("contains milk", "may contain nuts").
    allergen_statements: list[str] = Field(default_factory=list)
    # How legible the label was, 0..1, as judged by the extractor.
    legibility: float = Field(default=1.0, ge=0, le=1)
    source: Literal["image", "text", "barcode"] = "text"


class MatchedIngredient(BaseModel):
    original: str
    name: str
    code: str | None = None
    category: str
    risk: Literal["none", "low", "moderate", "high"]
    note: str = ""
    sources: list[str] = Field(default_factory=list)
    not_permitted_in: list[str] = Field(default_factory=list)
    # "db" = curated knowledge base; "ai" = classified by the LLM, unverified.
    origin: Literal["db", "ai"] = "db"


class Signal(BaseModel):
    """One reason that contributes to the verdict. Always traceable to data."""

    kind: Literal["nutrient", "ingredient", "allergen", "regulatory", "condition", "processing", "claim"]
    level: Level
    title: str
    detail: str
    evidence: str = ""  # which ingredient / nutrient value triggered it


class AnalysisResult(BaseModel):
    verdict: Verdict
    score: float  # 0 (best) .. 10 (worst)
    confidence: float
    label: LabelData
    ingredients: list[MatchedIngredient]
    unknown_ingredients: list[str]
    signals: list[Signal]
    positives: list[str] = Field(default_factory=list)
    # Allergens detected on the label: {"milk": "contains", "peanut": "may_contain"}
    allergens: dict[str, str] = Field(default_factory=dict)
    ultra_processed: bool
    explanation: str
    tips: list[str]
    warnings: list[str]
    profile: UserProfile = Field(default_factory=UserProfile)
    provider: str | None = None
    model: str | None = None
