"""CLI + sanitizer + musescore-wrapper tests (offline)."""

from typer.testing import CliRunner

from pdf2mscz.cli import app
from pdf2mscz.utils import musescore_cli
from pdf2mscz.utils.xml_sanitizer import is_valid_musicxml, sanitize_musicxml

runner = CliRunner()


def test_providers_command():
    r = runner.invoke(app, ["providers"])
    assert r.exit_code == 0
    assert "openai" in r.output


def test_check_deps_command():
    r = runner.invoke(app, ["check-deps"])
    assert r.exit_code == 0
    assert "MuseScore" in r.output


def test_sanitizer_strips_fences():
    dirty = 'Here you go:\n```xml\n<?xml version="1.0"?><score-partwise version="3.1"><part-list/><part id="P1"/></score-partwise>\n```'
    clean = sanitize_musicxml(dirty)
    assert "```" not in clean
    assert is_valid_musicxml(clean)


def test_sanitizer_fallback_on_garbage():
    clean = sanitize_musicxml("definitely not xml {{{")
    assert is_valid_musicxml(clean)


def test_musescore_missing_binary(monkeypatch):
    monkeypatch.setattr(musescore_cli.shutil, "which", lambda *_: None)
    assert musescore_cli.find_musescore() is None
    assert not musescore_cli.is_available()
