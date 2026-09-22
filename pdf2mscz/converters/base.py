"""Abstract provider interface + registry for OMR/VLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from PIL import Image
from pydantic import BaseModel, Field

from pdf2mscz.utils.prompts import REPAIR_PROMPT


class ConversionResult(BaseModel):
    """Normalized output of any provider."""

    musicxml: str = Field(description="Raw (ideally valid) MusicXML document")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    provider: str = Field(default="")
    model: str = Field(default="")
    truncated: bool = Field(
        default=False, description="Backend hit its output-token limit (finish_reason=length)."
    )


class ProviderConfig(BaseModel):
    """Common knobs passed to every provider."""

    model_config = {"extra": "allow"}

    model: str = ""
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    timeout: float = 120.0
    #: Output cap handed to the backend; ``0`` omits the parameter entirely.
    max_tokens: int = 8192


class AbstractProvider(ABC):
    """Base class for all OMR/VLM providers.

    Subclasses implement :meth:`image_to_musicxml`. Use
    :meth:`supports_refinement` to advertise whether the provider can
    correct third-party MusicXML given the source images.
    """

    name: ClassVar[str] = "base"
    requires_api_key: ClassVar[bool] = False
    env_var: ClassVar[str | None] = None
    #: Free-form completion exists (VLMs) — enables the repair loop.
    supports_repair: ClassVar[bool] = True
    #: Page images can be cut into staff systems (false for classical OMR).
    supports_chunking: ClassVar[bool] = True

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self.config = config or ProviderConfig()
        #: Set by :meth:`complete_text` when the backend truncated its answer.
        self.last_truncated: bool = False

    @abstractmethod
    def image_to_musicxml(self, images: list[Image.Image]) -> ConversionResult:
        """Transcribe page images to MusicXML."""
        raise NotImplementedError

    def complete_text(self, prompt: str, images: list[Image.Image], temperature: float = 0.0) -> str:
        """Run an arbitrary prompt against the images. Sets ``last_truncated``."""
        raise NotImplementedError(f"{self.name} does not support free-form completion")

    def repair_musicxml(
        self,
        images: list[Image.Image],
        bad_xml: str,
        reason: str = "",
        detail: str = "",
    ) -> ConversionResult:
        """Ask the model to redo output that failed validation."""
        if not self.supports_repair:
            raise NotImplementedError(f"{self.name} does not support repair")
        prompt = REPAIR_PROMPT.format(reason=reason or "invalid XML", detail=detail, previous=bad_xml)
        text = self.complete_text(prompt, images, temperature=0.3)
        return ConversionResult(
            musicxml=text,
            confidence=0.6,
            provider=self.name,
            model=self.config.model,
            truncated=self.last_truncated,
        )

    def refine_musicxml(self, images: list[Image.Image], candidate_xml: str) -> ConversionResult:
        """Optionally correct foreign OMR output. Default: not supported."""
        raise NotImplementedError(f"{self.name} does not support refinement")

    @classmethod
    def supports_refinement(cls) -> bool:
        return cls.refine_musicxml is not AbstractProvider.refine_musicxml

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(name={self.name!r})"


_REGISTRY: dict[str, type[AbstractProvider]] = {}


def register_provider(cls: type[AbstractProvider]) -> type[AbstractProvider]:
    """Class decorator registering a provider under ``cls.name``."""
    _REGISTRY[cls.name] = cls
    return cls


def get_provider_class(name: str) -> type[AbstractProvider]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = sorted(_REGISTRY) or "none registered"
        raise KeyError(f"Unknown provider {name!r}. Available: {available}") from exc


def available_providers() -> list[str]:
    return sorted(_REGISTRY)
