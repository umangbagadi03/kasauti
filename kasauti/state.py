import operator
from typing import Annotated, TypedDict

from .llm import Image
from .models import LabelData, MatchedIngredient, Signal, UserProfile


class GraphState(TypedDict, total=False):
    # inputs
    image: Image | None
    text: str | None
    barcode: str | None
    profile: UserProfile
    # extraction / lookup
    label: LabelData | None
    # analysis
    matched: list[MatchedIngredient]
    unknown: list[str]
    ai_matches: list[MatchedIngredient]
    enriched: bool
    allergens: dict[str, str]
    signals: list[Signal]
    positives: list[str]
    ultra_processed: bool
    # decision
    verdict: str
    score: float
    confidence: float
    # explanation
    explanation: str
    tips: list[str]
    # appended to by any node
    warnings: Annotated[list[str], operator.add]
