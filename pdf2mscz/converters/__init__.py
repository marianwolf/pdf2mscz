"""Provider registry re-exports (import side-effects register providers)."""

from pdf2mscz.converters import (  # noqa: F401 — registration side effects
    anthropic_provider,
    gemini_provider,
    oemer_provider,
    ollama_provider,
    openai_provider,
)
from pdf2mscz.converters.base import (
    AbstractProvider,
    ConversionResult,
    ProviderConfig,
    available_providers,
    get_provider_class,
    register_provider,
)

__all__ = [
    "AbstractProvider",
    "ConversionResult",
    "ProviderConfig",
    "available_providers",
    "get_provider_class",
    "register_provider",
]
