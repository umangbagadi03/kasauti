from ..config import DEFAULT_MODELS, PROVIDERS, settings
from .base import Image, LLMClient, LLMError

__all__ = ["Image", "LLMClient", "LLMError", "get_llm", "available_providers"]

_clients: dict[tuple[str, str], LLMClient] = {}


def get_llm(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Return a (cached) client. Provider/model fall back to LLM_PROVIDER / LLM_MODEL."""
    provider = (provider or settings.provider).lower()
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}")
    if model is None:
        model = settings.model if provider == settings.provider and settings.model else DEFAULT_MODELS[provider]

    key = (provider, model)
    if key not in _clients:
        _clients[key] = _build(provider, model)
    return _clients[key]


def _build(provider: str, model: str) -> LLMClient:
    s = settings
    if provider == "anthropic":
        from .anthropic_provider import AnthropicClient

        return AnthropicClient(model, s.anthropic_api_key, s.timeout_s)
    if provider == "gemini":
        from .gemini_provider import GeminiClient

        return GeminiClient(model, s.gemini_api_key, s.timeout_s)

    from .openai_provider import OpenAICompatibleClient

    if provider == "openai":
        return OpenAICompatibleClient("openai", model, s.openai_api_key, s.openai_base_url, s.timeout_s)
    base = s.ollama_base_url.rstrip("/")
    return OpenAICompatibleClient("ollama", model, None, f"{base}/v1", max(s.timeout_s, 180))


def available_providers() -> dict[str, dict]:
    """Which providers are configured — for /providers and the CLI."""
    s = settings
    return {
        "anthropic": {"configured": bool(s.anthropic_api_key), "default_model": DEFAULT_MODELS["anthropic"]},
        "gemini": {"configured": bool(s.gemini_api_key), "default_model": DEFAULT_MODELS["gemini"]},
        "openai": {"configured": bool(s.openai_api_key), "default_model": DEFAULT_MODELS["openai"]},
        "ollama": {"configured": True, "default_model": DEFAULT_MODELS["ollama"], "base_url": s.ollama_base_url},
    }
