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


def test_musescore_flatpak_staging_outside_home(monkeypatch, tmp_path):
    import subprocess
    from pathlib import Path

    monkeypatch.setattr(
        musescore_cli,
        "find_musescore_command",
        lambda *a, **k: ["flatpak", "run", "org.musescore.MuseScore"],
    )
    monkeypatch.setattr(musescore_cli, "_outside_home", lambda p: True)

    def fake_run(argv, **kwargs):
        out = Path(argv[-1])
        out.write_bytes(b"fake-mscz")

        class _P:
            returncode = 0
            stderr = ""
            stdout = ""

        return _P()

    monkeypatch.setattr(subprocess, "run", fake_run)
    src = tmp_path / "in.musicxml"
    src.write_text("<score-partwise/>")
    dest = tmp_path / "out.mscz"
    got = musescore_cli.musicxml_to_mscz(src, dest)
    assert got == dest
    assert dest.read_bytes() == b"fake-mscz"


def test_musescore_missing_binary(monkeypatch):
    monkeypatch.setattr(musescore_cli.shutil, "which", lambda *_: None)
    assert musescore_cli.find_musescore() is None
    assert musescore_cli.find_musescore_command() is None
    assert not musescore_cli.is_available()


def test_musescore_flatpak_detection(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        musescore_cli.shutil, "which", lambda n: "/usr/bin/flatpak" if n == "flatpak" else None
    )

    class _Proc:
        returncode = 0

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _Proc(),
    )
    cmd = musescore_cli.find_musescore_command()
    assert cmd == ["/usr/bin/flatpak", "run", "org.musescore.MuseScore"]
    assert musescore_cli.is_available()


def test_musescore_flatpak_not_installed(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        musescore_cli.shutil, "which", lambda n: "/usr/bin/flatpak" if n == "flatpak" else None
    )

    class _Proc:
        returncode = 1

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    assert musescore_cli.find_musescore_command() is None
