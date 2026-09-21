"""Google Gemini vision provider (via google-genai SDK)."""

from __future__ import annotations

import os

from PIL import Image

from pdf2mscz.converters.base import (
    AbstractProvider,
    ConversionResult,
    ProviderConfig,
    register_provider,
)
from pdf2mscz.utils.prompts import REFINEMENT_PROMPT, SYSTEM_PROMPT


@register_provider
class GeminiProvider(AbstractProvider):
    name = "gemini"
    requires_api_key = True
    env_var = "GOOGLE_API_KEY"
    default_model = "gemini-1.5-pro"

    def _client(self):
        api_key = (
            self.config.api_key or os.getenv(self.env_var or "") or os.getenv("GEMINI_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing Google API key. Set GOOGLE_API_KEY or pass --api-key.")
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                'Missing dependency "google-genai". Install with: pip install "pdf2mscz[gemini]"'
            ) from exc
        return genai.Client(api_key=api_key)

    def image_to_musicxml(self, images: list[Image.Image]) -> ConversionResult:
        client = self._client()
        model = self.config.model or self.default_model
        resp = client.models.generate_content(
            model=model,
            contents=[SYSTEM_PROMPT, *images],
        )
        return ConversionResult(
            musicxml=(resp.text or "").strip(),
            confidence=0.7,
            provider=self.name,
            model=model,
        )

    def refine_musicxml(self, images: list[Image.Image], candidate_xml: str) -> ConversionResult:
        client = self._client()
        model = self.config.model or self.default_model
        resp = client.models.generate_content(
            model=model,
            contents=[REFINEMENT_PROMPT, f"CANDIDATE:\n{candidate_xml}", *images],
        )
        return ConversionResult(
            musicxml=(resp.text or "").strip(),
            confidence=0.8,
            provider=self.name,
            model=model,
        )

    @classmethod
    def supports_refinement(cls) -> bool:
        return True


def build(config: ProviderConfig) -> GeminiProvider:
    return GeminiProvider(config)
