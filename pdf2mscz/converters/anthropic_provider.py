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

    def image_to_musicxml(self, images: list[Image.Image]) -> ConversionResult:
        client = self._client()
        model = self.config.model or self.default_model
        msg = client.messages.create(
            model=model,
            max_tokens=16000,
            temperature=self.config.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._blocks(images, "Transcribe.")}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        return ConversionResult(musicxml=text, confidence=0.7, provider=self.name, model=model)

    def refine_musicxml(self, images: list[Image.Image], candidate_xml: str) -> ConversionResult:
        client = self._client()
        model = self.config.model or self.default_model
        msg = client.messages.create(
            model=model,
            max_tokens=16000,
            temperature=0.0,
            system=REFINEMENT_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": self._blocks(images, f"CANDIDATE:\n{candidate_xml}"),
                }
            ],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        return ConversionResult(musicxml=text, confidence=0.8, provider=self.name, model=model)

    @classmethod
    def supports_refinement(cls) -> bool:
        return True


def build(config: ProviderConfig) -> AnthropicProvider:
    return AnthropicProvider(config)
