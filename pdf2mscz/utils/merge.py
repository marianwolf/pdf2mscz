"""Assemble per-staff-system MusicXML fragments into one score document."""

from __future__ import annotations

from lxml import etree

from pdf2mscz.utils.xml_sanitizer import as_document

_PART_LIST = (
    '<part-list><score-part id="P1"><part-name>{name}</part-name></score-part></part-list>'
)


def _parse(text: str) -> etree._Element | None:
    try:
        root = etree.fromstring(
            as_document(text).encode("utf-8"), parser=etree.XMLParser(recover=True, huge_tree=True)
        )
    except Exception:
        return None
    if root is None or root.tag != "score-partwise":
        return None
    return root


def _measures(root: etree._Element) -> list[etree._Element]:
    return list(root.xpath(".//*[local-name()='part']/*[local-name()='measure']"))


def _part_name(root: etree._Element) -> str:
    found = root.xpath(".//*[local-name()='score-part']/*[local-name()='part-name']")
    if found and found[0].text:
        return found[0].text.strip()
    return "Music"


def first_attributes(xml: str) -> str:
    """Serialise the first ``<attributes>`` block, for prompt context."""
    root = _parse(xml)
    if root is None:
        return ""
    found = root.xpath(".//*[local-name()='attributes']")
    if not found:
        return ""
    return etree.tostring(found[0], encoding="unicode").strip()


def merge_documents(parts: list[str]) -> str:
    """Merge fragments/documents into a single ``score-partwise`` document.

    Measures are renumbered 1..N in input order. Multi-staff input is
    flattened into one part (the pipelines that use this transcribe one
    staff system at a time).
    """
    measures: list[etree._Element] = []
    part_name = ""
    for part in parts:
        if not part or not part.strip():
            continue
        root = _parse(part)
        if root is None:
            continue
        if not part_name:
            part_name = _part_name(root)
        measures.extend(_measures(root))

    for i, measure in enumerate(measures, start=1):
        measure.set("number", str(i))

    body = "".join(
        etree.tostring(m, encoding="unicode") for m in measures
    )
    part_list = _PART_LIST.format(name=part_name or "Music")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<score-partwise version="3.1">\n{part_list}\n'
        f'<part id="P1">{body}</part>\n'
        "</score-partwise>"
    )
