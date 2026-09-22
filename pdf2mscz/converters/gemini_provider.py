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

    def _generate(self, prompt: str, images: list[Image.Image], temperature: float):
        client = self._client()
        model = self.config.model or self.default_model
        from google.genai import types

        config_kwargs: dict = {"temperature": temperature}
        if self.config.max_tokens > 0:
            config_kwargs["max_output_tokens"] = self.config.max_tokens
        resp = client.models.generate_content(
            model=model,
            contents=[prompt, *images],
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = (resp.text or "").strip()
        # Gemini reports output-cap hits as finish_reason=MAX_TOKENS.
        finish = ""
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            finish = str(getattr(candidates[0], "finish_reason", "") or "")
        self.last_truncated = "MAX_TOKENS" in finish.upper()
        return text, self.last_truncated

    def complete_text(self, prompt: str, images: list[Image.Image], temperature: float = 0.0) -> str:
        text, _ = self._generate(prompt, images, temperature)
        return text

    def image_to_musicxml(self, images: list[Image.Image]) -> ConversionResult:
        model = self.config.model or self.default_model
        text, truncated = self._generate(SYSTEM_PROMPT, images, self.config.temperature)
        return ConversionResult(
            musicxml=text, confidence=0.7, provider=self.name, model=model, truncated=truncated
        )

    def refine_musicxml(self, images: list[Image.Image], candidate_xml: str) -> ConversionResult:
        model = self.config.model or self.default_model
        text, truncated = self._generate(
            REFINEMENT_PROMPT + f"\n\nCANDIDATE:\n{candidate_xml}", images, 0.0
        )
        return ConversionResult(
            musicxml=text, confidence=0.8, provider=self.name, model=model, truncated=truncated
        )

    @classmethod
    def supports_refinement(cls) -> bool:
        return True


def build(config: ProviderConfig) -> GeminiProvider:
    return GeminiProvider(config)
