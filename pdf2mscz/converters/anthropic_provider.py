"""Anthropic Claude vision provider."""

from __future__ import annotations

import base64
import os

from PIL import Image

from pdf2mscz.converters._images import image_to_png_bytes
from pdf2mscz.converters.base import (
    AbstractProvider,
    ConversionResult,
    ProviderConfig,
    register_provider,
)
from pdf2mscz.utils.prompts import REFINEMENT_PROMPT, SYSTEM_PROMPT


@register_provider
class AnthropicProvider(AbstractProvider):
    name = "anthropic"
    requires_api_key = True
    env_var = "ANTHROPIC_API_KEY"
    default_model = "claude-3-5-sonnet-20241022"

    def _client(self):
        api_key = self.config.api_key or os.getenv(self.env_var or "")
        if not api_key:
            raise ValueError("Missing Anthropic API key. Set ANTHROPIC_API_KEY or pass --api-key.")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                'Missing dependency "anthropic". Install with: pip install "pdf2mscz[anthropic]"'
            ) from exc
        return Anthropic(api_key=api_key, timeout=self.config.timeout)

    def _blocks(self, images: list[Image.Image], text: str) -> list[dict]:
        blocks: list[dict] = []
        for img in images:
            b64 = base64.b64encode(image_to_png_bytes(img)).decode("ascii")
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                }
            )
        blocks.append({"type": "text", "text": text})
        return blocks

    def _complete(self, system: str, images: list[Image.Image], user_text: str, temperature: float):
        client = self._client()
        model = self.config.model or self.default_model
        kwargs: dict = {
            "model": model,
            "max_tokens": self.config.max_tokens or 4096,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": self._blocks(images, user_text)}],
        }
        msg = client.messages.create(**kwargs)
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        # Anthropic signals output-cap hits with stop_reason == "max_tokens".
        self.last_truncated = getattr(msg, "stop_reason", None) == "max_tokens"
        return text, self.last_truncated

    def complete_text(self, prompt: str, images: list[Image.Image], temperature: float = 0.0) -> str:
        text, _ = self._complete(prompt, images, "Transcribe.", temperature)
        return text

    def image_to_musicxml(self, images: list[Image.Image]) -> ConversionResult:
        model = self.config.model or self.default_model
        text, truncated = self._complete(
            SYSTEM_PROMPT, images, "Transcribe.", self.config.temperature
        )
        return ConversionResult(
            musicxml=text, confidence=0.7, provider=self.name, model=model, truncated=truncated
        )

    def refine_musicxml(self, images: list[Image.Image], candidate_xml: str) -> ConversionResult:
        model = self.config.model or self.default_model
        text, truncated = self._complete(
            REFINEMENT_PROMPT, images, f"CANDIDATE:\n{candidate_xml}", 0.0
        )
        return ConversionResult(
            musicxml=text, confidence=0.8, provider=self.name, model=model, truncated=truncated
        )

    @classmethod
    def supports_refinement(cls) -> bool:
        return True


def build(config: ProviderConfig) -> AnthropicProvider:
    return AnthropicProvider(config)
