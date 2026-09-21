"""Repair LLM-generated MusicXML: strip markdown, ensure parseability."""

from __future__ import annotations

import re

from lxml import etree

_FENCE_RE = re.compile(r"```(?:xml)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_DECL_RE = re.compile(r"<\?xml.*?\?>", re.DOTALL)

MINIMAL_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>
  <part id="P1"><measure number="1"><attributes><divisions>1</divisions><clef><sign>G</sign><line>2</line></clef><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>4</duration></note></measure></part>
</score-partwise>"""


def strip_markdown(text: str) -> str:
    """Remove ``` fences / language tags LLMs like to add."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    return "\n".join(lines).strip()


def sanitize_musicxml(raw: str) -> str:
    """Return parseable MusicXML; fall back to a minimal valid doc."""
    text = strip_markdown(raw).strip()
    if not text:
        return MINIMAL_TEMPLATE
    # Drop any prose before the XML declaration / root element.
    idx = text.find("<score-partwise")
    decl = _DECL_RE.search(text)
    if idx != -1:
        text = (decl.group(0) + "\n" if decl else "") + text[idx:]
    try:
        etree.fromstring(text.encode("utf-8"))
        return text
    except etree.XMLSyntaxError:
        pass
    # Try recovery via lenient parser, then re-serialize.
    try:
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(text.encode("utf-8"), parser=parser)
        if root is not None and root.tag == "score-partwise":
            return etree.tostring(root, encoding="unicode", xml_declaration=True)
    except Exception:
        pass
    return MINIMAL_TEMPLATE


def is_valid_musicxml(text: str) -> bool:
    try:
        root = etree.fromstring(text.encode("utf-8"))
        return root.tag == "score-partwise"
    except Exception:
        return False
