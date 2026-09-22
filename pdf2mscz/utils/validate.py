"""Structural checks on MusicXML: is there actually a score in there?

Kept separate from the sanitizer so the pipeline can compare what came out of
the model against what went in (measure counts, pitched notes) instead of
declaring victory on "well-formed XML".
"""

from __future__ import annotations

from dataclasses import dataclass

from lxml import etree


@dataclass
class ScoreStats:
    """Element counts of a ``score-partwise`` document."""

    measures: int = 0
    pitched_notes: int = 0
    rests: int = 0
    parts: int = 0
    title: str = ""

    def summary(self) -> str:
        return f"{self.measures} measures, {self.pitched_notes} pitched notes, {self.rests} rests"


def localname(node: etree._Element) -> str:
    """Element name without an optional namespace."""
    tag = node.tag
    if not isinstance(tag, str):  # comments / PIs
        return ""
    return tag.rsplit("}", 1)[-1]


def _count(root: etree._Element, xpath: str) -> int:
    try:
        return int(float(root.xpath(f"count({xpath})")))
    except (ValueError, TypeError, etree.XPathError):
        return 0


def score_stats(root: etree._Element | None) -> ScoreStats:
    """Count musical content of a parsed ``score-partwise`` element."""
    if root is None:
        return ScoreStats()
    return ScoreStats(
        measures=_count(root, ".//*[local-name()='part']/*[local-name()='measure']"),
        pitched_notes=_count(root, ".//*[local-name()='note'][*[local-name()='pitch']]"),
        rests=_count(root, ".//*[local-name()='note'][*[local-name()='rest']]"),
        parts=_count(root, ".//*[local-name()='part']"),
        title=_first_text(root, ".//*[local-name()='work']/*[local-name()='work-title']"),
    )


def validate_score(xml: str) -> ScoreStats:
    """Count musical content. Returns zeros for anything unparseable."""
    if not xml or not xml.strip():
        return ScoreStats()
    try:
        root = etree.fromstring(xml.encode("utf-8"), parser=etree.XMLParser(recover=True))
    except Exception:
        return ScoreStats()
    return score_stats(root)


def _first_text(root: etree._Element, xpath: str) -> str:
    try:
        found = root.xpath(xpath)
    except etree.XPathError:
        return ""
    if not found:
        return ""
    return (found[0].text or "").strip()


def is_meaningful(stats: ScoreStats) -> bool:
    """A usable transcription has at least one measure with a real note."""
    return stats.measures >= 1 and stats.pitched_notes >= 1
