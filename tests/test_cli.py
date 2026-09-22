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
    dirty = (
        'Here you go:\n```xml\n<?xml version="1.0"?><score-partwise version="3.1">'
        '<part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>'
        '<part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave>'
        "</pitch><duration>4</duration></note></measure></part></score-partwise>\n```"
    )
    res = sanitize_musicxml(dirty)
    assert res.ok, res.reason
    assert "```" not in res.xml
    assert is_valid_musicxml(res.xml)


def test_sanitizer_garbage_is_flagged_not_swallowed():
    """WP0: garbage must be reported, never silently replaced by a template."""
    res = sanitize_musicxml("definitely not xml {{{")
    assert not res.ok
    assert res.reason == "prose_only"
    assert not res.placeholder
    assert "definitely" in res.xml  # raw input kept for inspection


def test_sanitizer_placeholder_only_with_allow_empty():
    res = sanitize_musicxml("nope", allow_empty=True)
    assert not res.ok and res.placeholder
    assert is_valid_musicxml(res.xml)


def test_sanitizer_flags_model_prose_wrap():
    """A reasoning dump ending in a bare <measure> is a fragment, not a score."""
    res = sanitize_musicxml("**Step 1**: I see a bass clef.\n\n`<measure><note/></measure>`")
    assert not res.ok
    assert res.reason == "fragment"


def test_sanitizer_accepts_and_counts_real_music():
    doc = (
        '<?xml version="1.0" encoding="UTF-8"?><score-partwise version="3.1">'
        '<part-list><score-part id="P1"><part-name>Viola</part-name></score-part></part-list>'
        '<part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave>'
        "</pitch><duration>4</duration></note></measure>"
        '<measure number="2"><note><rest/><duration>4</duration></note></measure></part>'
        "</score-partwise>"
    )
    res = sanitize_musicxml(doc)
    assert res.ok, res.reason
    assert res.stats is not None
    assert res.stats.measures == 2
    assert res.stats.pitched_notes == 1


def test_sanitizer_recovers_truncated_document():
    """Cut-off output is salvaged (and marked) instead of crashing lxml."""
    full = (
        '<score-partwise version="3.1"><part-list><score-part id="P1">'
        "<part-name>M</part-name></score-part></part-list><part id='P1'>"
        '<measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch>'
        "<duration>4</duration></note></measure>"
        '<measure number="2"><note><pitch><step>E</step><octave>4</octave></pitch>'
        "<duration>4</duration></note></measure></part></score-partwise>"
    )
    res = sanitize_musicxml(full[: full.rindex("</measure>")])
    assert res.ok, res.reason
    assert res.recovered
    # lxml's recover parser re-closes the truncated tail, so both measures survive.
    assert res.stats is not None and res.stats.measures == 2


def test_sanitizer_cleans_model_noise():
    """Commentary, duplicated measures and nonsense clefs must not survive."""
    dirty = (
        '<score-partwise version="3.1"><part-list><score-part id="P1">'
        "<part-name>Viola</part-name></score-part></part-list><part id='P1'>"
        '<measure number="1">` element for each measure and include the necessary '
        "attributes and notes. Here is the output:\n"
        '<measure number="1"><attributes><divisions>4</divisions><clef>8</clef>'
        "<key><fifths>2</fifths></key>"
        "<time><beats>12</beats><beat-type>8</beat-type></time></attributes>"
        "<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration>"
        "</note></measure></measure></part></score-partwise>"
    )
    res = sanitize_musicxml(dirty)
    assert res.ok, res.reason
    assert "Here is the output" not in res.xml
    assert "<clef>8</clef>" not in res.xml  # dropped: no <sign> child
    assert res.stats is not None
    assert res.stats.measures == 1  # prose-carrier wrapper removed
    assert res.stats.pitched_notes == 1


def test_sanitizer_drops_notes_without_pitch_data():
    doc = (
        '<score-partwise version="3.1"><part-list><score-part id="P1">'
        "<part-name>M</part-name></score-part></part-list><part id='P1'>"
        '<measure number="1"><note><duration>4</duration></note>'
        "<note><pitch><step>G</step></pitch><duration>4</duration></note>"
        "<note><pitch><step>A</step><octave>4</octave></pitch><duration>4</duration>"
        "</note></measure></part></score-partwise>"
    )
    res = sanitize_musicxml(doc)
    assert res.ok, res.reason
    assert res.stats is not None and res.stats.pitched_notes == 1


def test_sanitizer_rejects_rests_only_score():
    doc = (
        '<score-partwise version="3.1"><part-list><score-part id="P1">'
        "<part-name>M</part-name></score-part></part-list>"
        '<part id="P1"><measure number="1"><note><rest/><duration>4</duration></note>'
        "</measure></part></score-partwise>"
    )
    res = sanitize_musicxml(doc)
    assert not res.ok
    assert res.reason == "no_pitched_notes"


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
