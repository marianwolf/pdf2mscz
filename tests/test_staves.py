"""Staff-system detection, barline estimate and measure merging (offline)."""

from PIL import Image, ImageDraw

from pdf2mscz.utils.merge import first_attributes, merge_documents
from pdf2mscz.utils.staves import count_barlines, find_systems
from pdf2mscz.utils.validate import is_meaningful, validate_score


def _page(systems: int = 3, barlines: int = 4, width: int = 800) -> Image.Image:
    """Synthetic engraving: N systems of 5 full-width lines + vertical strokes."""
    height = 100 + systems * 150
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for s in range(systems):
        top = 80 + s * 150
        for line in range(5):
            y = top + line * 6
            draw.line([(40, y), (width - 40, y)], fill="black", width=2)
        for b in range(barlines):
            x = 200 + b * 140
            draw.line([(x, top), (x, top + 24)], fill="black", width=3)
    return img


def test_find_systems_counts_staff_lines():
    systems = find_systems(_page(systems=3))
    assert len(systems) == 3
    tops = sorted(t for t, _ in systems)
    # Boxes are ordered top-to-bottom and do not overlap heavily.
    assert tops == sorted(set(tops))


def test_find_systems_ignores_titles():
    img = _page(systems=2)
    draw = ImageDraw.Draw(img)
    draw.text((40, 10), "Ich will besingen die Liebe des Herrn", fill="black")
    assert len(find_systems(img)) == 2


def test_find_systems_blank_page_returns_nothing():
    assert find_systems(Image.new("RGB", (400, 400), "white")) == []


def test_count_barlines_matches_drawing():
    img = _page(systems=3, barlines=4)
    systems = find_systems(img)
    assert count_barlines(img, systems) == 3 * 4


def test_count_barlines_unknown_on_blank():
    assert count_barlines(Image.new("RGB", (300, 300), "white"), []) is None


_FRAG_1 = (
    '<measure number="1"><attributes><divisions>4</divisions>'
    "<clef><sign>C</sign><line>3</line></clef></attributes>"
    "<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration></note>"
    "</measure><measure number='2'>"
    "<note><pitch><step>F</step><octave>4</octave></pitch><duration>4</duration></note>"
    "</measure>"
)
_FRAG_2 = (
    "<measure number='1'><note><pitch><step>A</step><octave>4</octave></pitch>"
    "<duration>4</duration></note></measure>"
)


def test_merge_renumbers_and_builds_root():
    merged = merge_documents([_FRAG_1, _FRAG_2])
    stats = validate_score(merged)
    assert stats.measures == 3
    assert stats.pitched_notes == 3
    assert stats.parts == 1
    assert "<part-list>" in merged
    # Measures are renumbered 1..N in order.
    assert 'number="1"' in merged and 'number="3"' in merged


def test_merge_skips_unusable_parts():
    merged = merge_documents(["", "total prose", _FRAG_2])
    assert validate_score(merged).measures == 1


def test_first_attributes_extraction():
    attrs = first_attributes(_FRAG_1)
    assert "<divisions>4</divisions>" in attrs
    assert first_attributes("no xml here") == ""


def test_validate_flags_empty_documents():
    assert not is_meaningful(validate_score(""))
    assert not is_meaningful(validate_score("<score-partwise/>"))
