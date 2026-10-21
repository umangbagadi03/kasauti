"""Runtime settings, read once from the environment (and .env)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROVIDERS = ("anthropic", "gemini", "openai", "ollama")

# Vision-capable defaults. Override with LLM_MODEL (or per request).
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-5-mini",
    "ollama": "llama3.2-vision",
}


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str | None
    anthropic_api_key: str | None
    gemini_api_key: str | None
    openai_api_key: str | None
    openai_base_url: str | None
    ollama_base_url: str
    timeout_s: float
    cache_path: Path | None
    default_region: str

    @classmethod
    def from_env(cls) -> "Settings":
        cache = os.getenv("KASAUTI_CACHE_PATH", ".kasauti_cache.sqlite")
        return cls(
            provider=os.getenv("LLM_PROVIDER", "gemini").lower(),
            model=os.getenv("LLM_MODEL") or None,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            timeout_s=float(os.getenv("LLM_TIMEOUT_S", "60")),
            cache_path=Path(cache) if cache.lower() not in ("", "off", "none") else None,
            default_region=os.getenv("KASAUTI_DEFAULT_REGION", "IN").upper(),
        )


settings = Settings.from_env()
