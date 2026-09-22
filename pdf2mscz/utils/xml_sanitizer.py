"""Repair LLM-generated MusicXML: strip markdown, ensure parseability.

The old behaviour was to silently substitute a placeholder score whenever the
model output was unusable, which produced "successful" runs whose output was a
single empty measure. That is now an explicit, inspectable failure:
:func:`sanitize_musicxml` returns a :class:`SanitizeResult` carrying ``ok`` and
a machine-readable ``reason``, and only substitutes the placeholder when the
caller opted in via ``allow_empty=True``.

Beyond parseability, model output is *cleaned*: commentary that leaked into
the markup is dropped, prose-wrapped duplicate measures are unwrapped, and
structurally impossible elements (``<clef>8</clef>`` and friends) are removed
so MuseScore can still import the result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lxml import etree

from pdf2mscz.utils.validate import ScoreStats, is_meaningful, localname, score_stats

_FENCE_RE = re.compile(r"```(?:xml)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_DECL_RE = re.compile(r"<\?xml.*?\?>", re.DOTALL)

MINIMAL_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>
  <part id="P1"><measure number="1"><attributes><divisions>1</divisions><clef><sign>G</sign><line>2</line></clef><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>4</duration></note></measure></part>
</score-partwise>"""

# Root-only document used to wrap bare `<measure>` fragments from the
# system-wise transcription mode.
_EMPTY_SHELL = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<score-partwise version="3.1">\n'
    '  <part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>\n'
    '  <part id="P1">\n{inner}\n  </part>\n'
    "</score-partwise>"
)

# Children that make a <measure> a real measure rather than a prose carrier.
_MEASURE_CONTENT = {"note", "backup", "forward", "harmony", "direction", "print"}
# <clef>/<key>/<time> must carry these children to be valid MusicXML.
_REQUIRED_CHILDREN = {"clef": {"sign"}, "key": {"fifths"}, "time": {"beats", "beat-type"}}


@dataclass
class SanitizeResult:
    """Outcome of turning raw model output into usable MusicXML."""

    xml: str
    ok: bool
    reason: str = ""  # empty when ok; one of the REASONS below otherwise
    recovered: bool = False  # salvaged by lxml's lenient parser
    placeholder: bool = False  # MINIMAL_TEMPLATE was substituted (allow_empty)
    detail: str = ""  # parser message, for error reports
    stats: ScoreStats | None = None  # filled in for accepted documents


# Machine-readable failure reasons (kept stable: the CLI prints them).
REASONS = ("empty", "prose_only", "fragment", "truncated", "invalid", "no_pitched_notes")


def strip_markdown(text: str) -> str:
    """Remove ``` fences / language tags LLMs like to add."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    return "\n".join(lines).strip()


def as_document(text: str) -> str:
    """Wrap a bare ``<measure>`` fragment in a minimal ``score-partwise`` doc.

    Full documents and non-XML prose are returned unchanged so that
    :func:`sanitize_musicxml` can classify them.
    """
    t = (text or "").strip()
    if "<score-partwise" in t:
        return t
    if "<measure" not in t:
        return t
    return _EMPTY_SHELL.format(inner=_DECL_RE.sub("", t).strip())


def _fail(text: str, reason: str, allow_empty: bool, detail: str = "") -> SanitizeResult:
    if allow_empty:
        return SanitizeResult(
            xml=MINIMAL_TEMPLATE, ok=False, reason=reason, placeholder=True, detail=detail
        )
    return SanitizeResult(xml=text, ok=False, reason=reason, detail=detail)


def sanitize_musicxml(raw: str, *, allow_empty: bool = False) -> SanitizeResult:
    """Return cleaned MusicXML plus an explicit success/failure verdict.

    ``ok=True`` means the document is real, non-placeholder MusicXML the
    pipeline may write out. ``allow_empty=True`` substitutes
    :data:`MINIMAL_TEMPLATE` on failure (``placeholder=True``) instead of
    returning the unusable input.
    """
    text = strip_markdown(raw or "").strip()
    if not text:
        return _fail(text, "empty", allow_empty)

    idx = text.find("<score-partwise")
    if idx == -1:
        reason = "fragment" if ("<measure" in text or "<note" in text) else "prose_only"
        return _fail(text, reason, allow_empty)

    decl = _DECL_RE.search(text)
    body = (decl.group(0) + "\n" if decl else "") + text[idx:]

    root, detail, strict = _parse(body)
    if root is None:
        reason = "truncated" if "</score-partwise>" not in body else "invalid"
        return _fail(body, reason, allow_empty, detail)

    clean(root)
    stats = score_stats(root)
    if not is_meaningful(stats):
        reason = "truncated" if "</score-partwise>" not in body else "no_pitched_notes"
        return _fail(body, reason, allow_empty, detail)

    xml = etree.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
    return SanitizeResult(xml=xml, ok=True, recovered=not strict, detail=detail, stats=stats)


def _parse(body: str) -> tuple[etree._Element | None, str, bool]:
    """Strict parse first, then lxml's lenient parser. ``(root, detail, strict)``."""
    try:
        etree.fromstring(body.encode("utf-8"))
    except etree.XMLSyntaxError as exc:
        detail = str(exc).splitlines()[0]
        try:
            root = etree.fromstring(body.encode("utf-8"), parser=etree.XMLParser(recover=True))
        except Exception:
            return None, detail, False
        if root is None or root.tag != "score-partwise":
            return None, detail, False
        return root, detail, False
    try:
        root = etree.fromstring(body.encode("utf-8"))
    except etree.XMLSyntaxError:  # pragma: no cover - just parsed above
        return None, "", False
    return root, "", True


def clean(root: etree._Element) -> None:
    """Normalise model output: drop prose, un-duplicate measures, fix shapes."""
    _drop_prose(root)
    _unwrap_nested_measures(root)
    _drop_empty_measures(root)
    _drop_malformed(root)
    _drop_empty_parts(root)


def _drop_prose(root: etree._Element) -> None:
    """Remove commentary the model interleaved with the markup.

    Valid MusicXML never carries free text as an element's value when the
    element has children, nor as a tail between elements.
    """
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        if len(node) and node.text and node.text.strip():
            node.text = None
        if node.tail and node.tail.strip():
            node.tail = None


def _unwrap_nested_measures(root: etree._Element) -> None:
    """Flatten ``<measure>…<measure>`` duplication caused by prose wrapping."""
    for _ in range(5):
        moved = False
        for measure in list(root.iter()):
            if localname(measure) != "measure":
                continue
            parent = measure.getparent()
            if parent is None:
                continue
            nested = [c for c in measure if isinstance(c.tag, str) and localname(c) == "measure"]
            for child in nested:
                measure.remove(child)
                parent.insert(list(parent).index(measure) + 1, child)
                moved = True
        if not moved:
            return


def _drop_empty_measures(root: etree._Element) -> None:
    """Delete measures holding no musical content (prose carriers)."""
    for part in [n for n in root.iter() if localname(n) == "part"]:
        for measure in list(part):
            if localname(measure) != "measure":
                continue
            if not any(localname(c) in _MEASURE_CONTENT for c in measure):
                part.remove(measure)


def _drop_malformed(root: etree._Element) -> None:
    """Remove elements whose required children are missing or nonsensical."""
    for node in list(root.iter()):
        name = localname(node)
        if name in _REQUIRED_CHILDREN:
            kids = {localname(c) for c in node}
            missing = _REQUIRED_CHILDREN[name] - kids
            if missing:
                _remove(node)
        elif name == "divisions":
            if not (node.text or "").strip().isdigit():
                _remove(node)
        elif name == "pitch":
            kids = {localname(c) for c in node}
            if not {"step", "octave"} <= kids:
                # A pitch without step/octave invalidates its whole note.
                note = node.getparent()
                _remove(note if note is not None and localname(note) == "note" else node)


def _remove(node: etree._Element | None) -> None:
    parent = node.getparent() if node is not None else None
    if parent is not None:
        parent.remove(node)


def _drop_empty_parts(root: etree._Element) -> None:
    for part in [n for n in root.iter() if localname(n) == "part"]:
        if not any(localname(c) == "measure" for c in part):
            _remove(part)


def is_valid_musicxml(text: str) -> bool:
    try:
        root = etree.fromstring((text or "").encode("utf-8"))
    except etree.XMLSyntaxError:
        return False
    return root.tag == "score-partwise"
