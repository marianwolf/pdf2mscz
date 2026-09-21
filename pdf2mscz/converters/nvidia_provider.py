"""NVIDIA NIM vision provider (build.nvidia.com, OpenAI-compatible API)."""

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

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


@register_provider
class NvidiaProvider(AbstractProvider):
    name = "nvidia"
    requires_api_key = True
    env_var = "NVIDIA_API_KEY"
    default_model = "meta/llama-3.2-90b-vision-instruct"

    def _client(self):  # lazy import so package works without the SDK
        api_key = self.config.api_key or os.getenv(self.env_var or "")
        if not api_key:
            raise ValueError("Missing NVIDIA API key. Set NVIDIA_API_KEY or pass --api-key.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                'Missing dependency "openai". Install with: pip install "pdf2mscz[nvidia]"'
            ) from exc
        # NIM exposes an OpenAI-compatible endpoint, so reuse the OpenAI SDK.
        return OpenAI(
            api_key=api_key,
            base_url=self.config.base_url or NIM_BASE_URL,
            timeout=self.config.timeout,
        )

    def _complete(self, content: list[dict], temperature: float) -> str:
        client = self._client()
        model = self.config.model or self.default_model
        # No max_tokens: output caps differ per NIM model (e.g. phi-3-vision
        # allows 4096), and omitting it lets the server pick the model limit.
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],  # type: ignore[arg-type]
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    @staticmethod
    def _page_content(prompt: str, images: list[Image.Image]) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})
        return content

    def image_to_musicxml(self, images: list[Image.Image]) -> ConversionResult:
        model = self.config.model or self.default_model
        text = self._complete(self._page_content(SYSTEM_PROMPT, images), self.config.temperature)
        return ConversionResult(musicxml=text, confidence=0.7, provider=self.name, model=model)

    def refine_musicxml(self, images: list[Image.Image], candidate_xml: str) -> ConversionResult:
        model = self.config.model or self.default_model
        prompt = REFINEMENT_PROMPT + "\n\nCANDIDATE:\n" + candidate_xml
        text = self._complete(self._page_content(prompt, images), 0.0)
        return ConversionResult(musicxml=text, confidence=0.8, provider=self.name, model=model)

    @classmethod
    def supports_refinement(cls) -> bool:
        return True


def build(config: ProviderConfig) -> NvidiaProvider:
    return NvidiaProvider(config)
