"""Abstract provider interface + registry for OMR/VLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from PIL import Image
from pydantic import BaseModel, Field


class ConversionResult(BaseModel):
    """Normalized output of any provider."""

    musicxml: str = Field(description="Raw (ideally valid) MusicXML document")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    provider: str = Field(default="")
    model: str = Field(default="")


class ProviderConfig(BaseModel):
    """Common knobs passed to every provider."""

    model_config = {"extra": "allow"}

    model: str = ""
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    timeout: float = 120.0


class AbstractProvider(ABC):
    """Base class for all OMR/VLM providers.

    Subclasses implement :meth:`image_to_musicxml`. Use
    :meth:`supports_refinement` to advertise whether the provider can
    correct third-party MusicXML given the source images.
    """

    name: ClassVar[str] = "base"
    requires_api_key: ClassVar[bool] = False
    env_var: ClassVar[str | None] = None

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self.config = config or ProviderConfig()

    @abstractmethod
    def image_to_musicxml(self, images: list[Image.Image]) -> ConversionResult:
        """Transcribe page images to MusicXML."""
        raise NotImplementedError

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
