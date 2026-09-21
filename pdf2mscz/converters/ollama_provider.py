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

    def image_to_musicxml(self, images: list[Image.Image]) -> ConversionResult:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover — httpx is a core dep
            raise ImportError(
                'Missing dependency "httpx". Install with: pip install "pdf2mscz[ollama]"'
            ) from exc

        base_url = (self.config.base_url or "http://localhost:11434").rstrip("/")
        model = self.config.model or self.default_model
        content: list[dict] = [{"type": "text", "text": SYSTEM_PROMPT}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})
        # Ollama exposes an OpenAI-compatible endpoint at /v1/chat/completions.
        r = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.config.temperature,
            },
            timeout=self.config.timeout,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        return ConversionResult(musicxml=text, confidence=0.5, provider=self.name, model=model)


def build(config: ProviderConfig) -> OllamaProvider:
    return OllamaProvider(config)
