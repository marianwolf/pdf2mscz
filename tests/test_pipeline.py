"""Pipeline behaviour: no silent placeholder scores, retries, chunking (offline)."""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from pdf2mscz.converters.base import (
    AbstractProvider,
    ConversionResult,
    ProviderConfig,
    get_provider_class,
    register_provider,
)
from pdf2mscz.pipeline import ConversionOptions, TranscriptionError, convert

VALID_DOC = (
    '<?xml version="1.0" encoding="UTF-8"?><score-partwise version="3.1">'
    '<part-list><score-part id="P1"><part-name>Viola</part-name></score-part></part-list>'
    '<part id="P1"><measure number="1"><attributes><divisions>4</divisions>'
    "<clef><sign>C</sign><line>3</line></clef></attributes>"
    "<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration></note>"
    "</measure><measure number='2'>"
    "<note><pitch><step>F</step><octave>4</octave></pitch><duration>4</duration></note>"
    "</measure></part></score-partwise>"
)


def _provider(name: str, answers: list[str]) -> type[AbstractProvider]:
    """Register a provider that walks through ``answers`` (last one repeats).

    Built via ``type()`` so ``register_provider`` sees the intended name and
    per-class call counter.
    """

    def complete_text(self, prompt, images, temperature=0.0):
        cls = type(self)
        idx = min(cls.calls, len(answers) - 1)
        cls.calls += 1
        self.last_truncated = False
        return answers[idx]

    def image_to_musicxml(self, images):  # pragma: no cover - pipeline uses complete_text
        return ConversionResult(musicxml=answers[0], provider=name)

    cls = type(
        f"FakeProvider_{name.replace('-', '_')}",
        (AbstractProvider,),
        {
            "name": name,
            "calls": 0,
            "complete_text": complete_text,
            "image_to_musicxml": image_to_musicxml,
        },
    )
    register_provider(cls)
    return cls


def _opts(tmp_path: Path, provider: str, **kwargs) -> ConversionOptions:
    defaults = {"provider": provider, "chunk": "page", "output_format": "musicxml", "dpi": 72}
    defaults.update(kwargs)
    return ConversionOptions(**defaults)


def test_garbage_output_raises_and_writes_no_score(tmp_path):
    _provider("fake-garbage", ["I think the clef is a bass clef... no XML at all."])
    src = tmp_path / "in.png"
    Image.new("RGB", (60, 60), "white").save(src)
    out = tmp_path / "score.mscz"

    with pytest.raises(TranscriptionError) as excinfo:
        convert(src, out, _opts(tmp_path, "fake-garbage"))

    assert excinfo.value.reason == "prose_only"
    assert not (tmp_path / "score.musicxml").exists()  # nothing written
    assert not (tmp_path / "score.mscz").exists()
    # WP0: the raw model answer is preserved for inspection.
    raw = tmp_path / "score.raw.txt"
    assert raw.exists()
    assert "bass clef" in raw.read_text(encoding="utf-8")


def test_empty_output_raises_with_reason(tmp_path):
    _provider("fake-empty", [""])
    src = tmp_path / "in.png"
    Image.new("RGB", (60, 60), "white").save(src)

    with pytest.raises(TranscriptionError) as excinfo:
        convert(src, tmp_path / "score.mscz", _opts(tmp_path, "fake-empty"))
    assert excinfo.value.reason == "empty"


def test_valid_output_writes_musicxml_and_stats(tmp_path):
    _provider("fake-good", [VALID_DOC])
    src = tmp_path / "in.png"
    Image.new("RGB", (60, 60), "white").save(src)
    out = tmp_path / "score.mscz"

    result = convert(src, out, _opts(tmp_path, "fake-good"))

    assert result.musicxml_path.exists()
    assert result.stats is not None
    assert result.stats.measures == 2
    assert result.stats.pitched_notes == 2
    assert "score-partwise" in result.musicxml_path.read_text(encoding="utf-8")


def test_repair_loop_recovers_from_bad_first_answer(tmp_path):
    """WP1: a bad first answer is repaired instead of failing outright."""
    cls = _provider("fake-repair", ["totally broken", VALID_DOC])
    src = tmp_path / "in.png"
    Image.new("RGB", (60, 60), "white").save(src)

    result = convert(src, tmp_path / "score.mscz", _opts(tmp_path, "fake-repair", retry=1))

    assert cls.calls == 2  # first answer + one repair round
    assert result.stats is not None and result.stats.measures == 2


def test_no_retry_gives_up_immediately(tmp_path):
    cls = _provider("fake-norepair", ["broken", VALID_DOC])
    src = tmp_path / "in.png"
    Image.new("RGB", (60, 60), "white").save(src)

    with pytest.raises(TranscriptionError):
        convert(src, tmp_path / "score.mscz", _opts(tmp_path, "fake-norepair", retry=0))
    assert cls.calls == 1


def test_allow_empty_writes_placeholder_with_warning(tmp_path):
    _provider("fake-placeholder", ["nope"])
    src = tmp_path / "in.png"
    Image.new("RGB", (60, 60), "white").save(src)

    result = convert(
        src, tmp_path / "score.mscz", _opts(tmp_path, "fake-placeholder", allow_empty=True)
    )
    assert result.musicxml_path.exists()
    assert any("placeholder" in w for w in result.warnings)


def test_measure_shortfall_is_reported(tmp_path):
    """WP3-lite: warn when far fewer measures than printed barlines come back."""
    _provider("fake-short", [VALID_DOC])  # 2 measures

    img = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(img)
    for system_y in (80, 230):
        for line in range(5):
            y = system_y + line * 6
            draw.line([(40, y), (760, y)], fill="black", width=2)
        for x in (200, 340, 480, 620, 760 - 40):
            draw.line([(x, system_y), (x, system_y + 24)], fill="black", width=3)
    src = tmp_path / "in.png"
    img.save(src)

    result = convert(
        src,
        tmp_path / "score.mscz",
        _opts(tmp_path, "fake-short", chunk="system"),
    )
    assert any("measures" in w for w in result.warnings), result.warnings


def test_system_chunking_calls_provider_per_system(tmp_path):
    """WP2: one request per staff system, merged into a single score."""
    cls = _provider("fake-chunk", [VALID_DOC])

    img = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(img)
    for system_y in (80, 230):
        for line in range(5):
            y = system_y + line * 6
            draw.line([(40, y), (760, y)], fill="black", width=2)
    src = tmp_path / "in.png"
    img.save(src)

    result = convert(
        src, tmp_path / "score.mscz", _opts(tmp_path, "fake-chunk", chunk="system")
    )
    assert cls.calls == 2  # two systems → two requests
    # Each fragment contributes its own measures, merged and renumbered.
    assert result.stats is not None
    assert result.stats.measures == 4


def test_provider_without_chunking_is_left_alone(tmp_path):
    """Classical OMR (oemer) must not be prompted per staff system."""
    from pdf2mscz.converters.oemer_provider import OemerProvider

    assert OemerProvider.supports_chunking is False
    assert OemerProvider.supports_repair is False
    assert get_provider_class("oemer").supports_chunking is False


def test_raw_output_disabled_when_requested(tmp_path):
    _provider("fake-noraw", ["broken prose"])
    src = tmp_path / "in.png"
    Image.new("RGB", (60, 60), "white").save(src)

    with pytest.raises(TranscriptionError) as excinfo:
        convert(src, tmp_path / "score.mscz", _opts(tmp_path, "fake-noraw", save_raw=False))
    assert excinfo.value.raw_path is None
    assert not (tmp_path / "score.raw.txt").exists()


def test_provider_config_carries_max_tokens():
    cfg = ProviderConfig(max_tokens=2048)
    assert cfg.max_tokens == 2048
    assert ProviderConfig().max_tokens > 0
