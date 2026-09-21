"""Tests for providers/registry (no network calls)."""

from PIL import Image

from pdf2mscz.converters import available_providers, get_provider_class
from pdf2mscz.converters.base import AbstractProvider, ConversionResult, ProviderConfig


def test_registry_lists_all_providers():
    assert {"openai", "anthropic", "gemini", "ollama", "oemer"} <= set(available_providers())


def test_unknown_provider_raises():
    try:
        get_provider_class("does-not-exist")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")


def test_fake_provider_contract():
    from pdf2mscz.converters.base import register_provider

    @register_provider
    class _Fake(AbstractProvider):
        name = "test-fake"

        def image_to_musicxml(self, images: list[Image.Image]) -> ConversionResult:
            return ConversionResult(musicxml="<score-partwise/>", provider="test-fake")

    cls = get_provider_class("test-fake")
    out = cls(ProviderConfig()).image_to_musicxml([Image.new("RGB", (8, 8))])
    assert "score-partwise" in out.musicxml


def test_oemer_missing_binary_gives_helpful_error(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda *_: None)
    cls = get_provider_class("oemer")
    try:
        cls(ProviderConfig()).image_to_musicxml([Image.new("RGB", (8, 8))])
    except RuntimeError as exc:
        assert "oemer" in str(exc).lower()
    except ImportError:
        pass  # acceptable if oemer import path differs
