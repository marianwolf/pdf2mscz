"""Repair LLM-generated MusicXML: strip markdown, ensure parseability.

The old behaviour was to silently substitute a placeholder score whenever the
model output was unusable, which produced "successful" runs whose output was a
single empty measure. That is now an explicit, inspectable failure:
:func:`sanitize_musicxml` returns a :class:`SanitizeResult` carrying ``ok`` and
a machine-readable ``reason``, and only substitutes the placeholder when the
caller opted in via ``allow_empty=True``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lxml import etree

from pdf2mscz.utils.validate import ScoreStats, is_meaningful, validate_score

_FENCE_RE = re.compile(r"```(?:xml)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_DECL_RE = re.compile(r"<\?xml.*?\?>", re.DOTALL)

ROOT_OPEN = '<score-partwise version="3.1">'

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
    """Return parseable MusicXML plus an explicit success/failure verdict.

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

    try:
        etree.fromstring(body.encode("utf-8"))
    except etree.XMLSyntaxError as exc:
        detail = str(exc).splitlines()[0]
        recovered = _recover(body)
        if recovered is None:
            reason = "truncated" if "</score-partwise>" not in body else "invalid"
            return _fail(body, reason, allow_empty, detail)
        stats = validate_score(recovered)
        if not is_meaningful(stats):
            reason = "truncated" if "</score-partwise>" not in body else "no_pitched_notes"
            return _fail(body, reason, allow_empty, detail)
        # Salvageable: keep it, but tell the caller it was cut off / repaired.
        return SanitizeResult(xml=recovered, ok=True, recovered=True, detail=detail, stats=stats)
    else:
        stats = validate_score(body)
        if not is_meaningful(stats):
            return _fail(body, "no_pitched_notes", allow_empty)
        return SanitizeResult(xml=body, ok=True, stats=stats)


def _recover(body: str) -> str | None:
    """Lenient re-parse; return XML text or ``None`` when unusable."""
    try:
        root = etree.fromstring(body.encode("utf-8"), parser=etree.XMLParser(recover=True))
    except Exception:
        return None
    if root is None or root.tag != "score-partwise":
        return None
    # `xml_declaration=True` is only valid for byte encodings (lxml raises
    # "Serialisation to unicode must not request an XML declaration").
    return etree.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")


def is_valid_musicxml(text: str) -> bool:
    try:
        root = etree.fromstring((text or "").encode("utf-8"))
    except etree.XMLSyntaxError:
        return False
    return root.tag == "score-partwise"
