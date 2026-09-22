"""Ollama / local VLM provider (OpenAI-compatible chat API)."""

from __future__ import annotations

from PIL import Image

from pdf2mscz.converters._images import image_to_data_url
from pdf2mscz.converters.base import (
    AbstractProvider,
    ConversionResult,
    ProviderConfig,
    register_provider,
)
from pdf2mscz.utils.prompts import SYSTEM_PROMPT


@register_provider
class OllamaProvider(AbstractProvider):
    name = "ollama"
    requires_api_key = False
    env_var = None
    default_model = "llava:13b"

    def _complete(self, prompt: str, images: list[Image.Image], temperature: float) -> tuple[str, bool]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover — httpx is a core dep
            raise ImportError(
                'Missing dependency "httpx". Install with: pip install "pdf2mscz[ollama]"'
            ) from exc

        base_url = (self.config.base_url or "http://localhost:11434").rstrip("/")
        model = self.config.model or self.default_model
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})
        body: dict = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
        }
        if self.config.max_tokens > 0:
            body["options"] = {"num_predict": self.config.max_tokens}
        # Ollama exposes an OpenAI-compatible endpoint at /v1/chat/completions.
        r = httpx.post(f"{base_url}/v1/chat/completions", json=body, timeout=self.config.timeout)
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        self.last_truncated = choice.get("finish_reason") == "length"
        return choice["message"]["content"].strip(), self.last_truncated

    def complete_text(self, prompt: str, images: list[Image.Image], temperature: float = 0.0) -> str:
        text, _ = self._complete(prompt, images, temperature)
        return text

    def image_to_musicxml(self, images: list[Image.Image]) -> ConversionResult:
        model = self.config.model or self.default_model
        text, truncated = self._complete(SYSTEM_PROMPT, images, self.config.temperature)
        return ConversionResult(
            musicxml=text, confidence=0.5, provider=self.name, model=model, truncated=truncated
        )


def build(config: ProviderConfig) -> OllamaProvider:
    return OllamaProvider(config)
