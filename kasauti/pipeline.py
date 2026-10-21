"""Public entry point: analyze(...) -> AnalysisResult."""

from .graph import build_graph
from .images import prepare_image
from .llm import LLMClient, get_llm
from .models import AnalysisResult, LabelData, UserProfile


def analyze(
    *,
    image: bytes | None = None,
    text: str | None = None,
    barcode: str | None = None,
    profile: UserProfile | None = None,
    provider: str | None = None,
    model: str | None = None,
    use_llm: bool = True,
    llm: LLMClient | None = None,
) -> AnalysisResult:
    """Analyse a label from a photo, pasted text and/or a barcode.

    use_llm=False runs fully offline (text or barcode input only, template explanation).
    Pass `llm` to inject a client directly (tests, custom providers).
    """
    if not (image or text or barcode):
        raise ValueError("Provide an image, text or barcode.")
    if llm is None and use_llm:
        llm = get_llm(provider, model)  # raises LLMError with a clear message if misconfigured

    state = build_graph().invoke(
        {
            "image": prepare_image(image) if image else None,
            "text": text,
            "barcode": barcode,
            "profile": profile or UserProfile(),
            "warnings": [],
        },
        config={"configurable": {"llm": llm}, "run_name": "kasauti"},
    )

    return AnalysisResult(
        verdict=state.get("verdict", "Unknown"),
        score=state.get("score", 0.0),
        confidence=state.get("confidence", 0.0),
        label=state.get("label") or LabelData(),
        ingredients=state.get("matched", []),
        unknown_ingredients=state.get("unknown", []),
        signals=state.get("signals", []),
        positives=state.get("positives", []),
        allergens=state.get("allergens", {}),
        ultra_processed=state.get("ultra_processed", False),
        explanation=state.get("explanation", ""),
        tips=state.get("tips", []),
        warnings=list(dict.fromkeys(state.get("warnings", []))),
        profile=state.get("profile") or UserProfile(),
        provider=llm.provider if llm else None,
        model=llm.model if llm else None,
    )
