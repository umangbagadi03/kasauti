import base64

from .base import Image, LLMClient, LLMError

MAX_TOKENS = 4096


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, api_key: str | None, timeout: float):
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)

    def _content(self, prompt: str, image: Image | None) -> list[dict]:
        blocks: list[dict] = []
        if image:
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.mime_type,
                    "data": base64.b64encode(image.data).decode(),
                },
            })
        blocks.append({"type": "text", "text": prompt})
        return blocks

    def _text(self, system, prompt, image):
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": self._content(prompt, image)}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    def _json(self, system, prompt, schema, image):
        # Forcing a tool call makes Claude return arguments that follow the schema.
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=[{"name": "submit", "description": "Submit the structured answer.", "input_schema": schema}],
            tool_choice={"type": "tool", "name": "submit"},
            messages=[{"role": "user", "content": self._content(prompt, image)}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return block.input
        raise ValueError("model did not return a tool call")
