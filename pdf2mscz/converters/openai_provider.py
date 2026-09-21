"""OpenAI GPT-4o(-mini) vision provider."""

from __future__ import annotations

import os

from PIL import Image

from pdf2mscz.converters._images import image_to_data_url
from pdf2mscz.converters.base import (
    AbstractProvider,
    ConversionResult,
    ProviderConfig,
    register_provider,
)
from pdf2mscz.utils.prompts import REFINEMENT_PROMPT, SYSTEM_PROMPT


@register_provider
class OpenAIProvider(AbstractProvider):
    name = "openai"
    requires_api_key = True
    env_var = "OPENAI_API_KEY"
    default_model = "gpt-4o"

    def _client(self):  # lazy import so package works without the SDK
        api_key = self.config.api_key or os.getenv(self.env_var or "")
        if not api_key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or pass --api-key.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                'Missing dependency "openai". Install with: pip install "pdf2mscz[openai]"'
            ) from exc
        return OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    def image_to_musicxml(self, images: list[Image.Image]) -> ConversionResult:
        client = self._client()
        model = self.config.model or self.default_model
        content: list[dict] = [{"type": "text", "text": SYSTEM_PROMPT}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],  # type: ignore[arg-type]
            temperature=self.config.temperature,
            max_tokens=16000,
        )
        text = (resp.choices[0].message.content or "").strip()
        return ConversionResult(musicxml=text, confidence=0.7, provider=self.name, model=model)

    def refine_musicxml(self, images: list[Image.Image], candidate_xml: str) -> ConversionResult:
        client = self._client()
        model = self.config.model or self.default_model
        content: list[dict] = [
            {"type": "text", "text": REFINEMENT_PROMPT + "\n\nCANDIDATE:\n" + candidate_xml}
        ]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],  # type: ignore[arg-type]
            temperature=0.0,
            max_tokens=16000,
        )
        text = (resp.choices[0].message.content or "").strip()
        return ConversionResult(musicxml=text, confidence=0.8, provider=self.name, model=model)

    @classmethod
    def supports_refinement(cls) -> bool:
        return True


def build(config: ProviderConfig) -> OpenAIProvider:
    return OpenAIProvider(config)
