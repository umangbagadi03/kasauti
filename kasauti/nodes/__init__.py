from langchain_core.runnables import RunnableConfig

from ..llm import LLMClient


def llm_from(config: RunnableConfig | None) -> LLMClient | None:
    """The LLM client is passed per run via config, so tests can inject a fake one."""
    return ((config or {}).get("configurable") or {}).get("llm")
