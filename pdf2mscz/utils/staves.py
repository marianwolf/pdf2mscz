"""Staff-system detection and barline estimation via horizontal/vertical ink projections.

Used to cut a rendered page into one image per staff system (each is a much
easier target for a VLM than a whole dense page) and to sanity-check the
transcription afterwards by comparing detected measures against the barlines
that are actually printed.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

#: ``(top, bottom)`` pixel rows of one staff system, bottom exclusive.
SystemBox = tuple[int, int]

# Tunables, chosen against a 300 dpi engraving (test.pdf) and synthetic images.
_INK_THRESHOLD = 200  # grayscale below this counts as ink
_MIN_INK_FRAC = 0.02  # row counts as "active" above this fraction of width
_STAFF_ROW_FRAC = 0.45  # a staff line spans at least this fraction of the width
_MERGE_GAP_FRAC = 0.018  # merge ink bands separated by less than this page height
_MARGIN_FRAC = 0.045  # crop margin for dynamics/measure numbers around a system
_BARLINE_FRAC = 0.95  # a barline fills this share of the staff height


def _runs(mask: np.ndarray) -> list[SystemBox]:
    """Contiguous ``True`` spans of a 1-D boolean array as (start, end)."""
    boxes: list[SystemBox] = []
    start: int | None = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            boxes.append((start, i))
            start = None
    if start is not None:
        boxes.append((start, len(mask)))
    return boxes


def _merge(bands: list[SystemBox], gap: int) -> list[SystemBox]:
    """Join bands whose separating blank rows are smaller than ``gap``."""
    if not bands:
        return []
    out = [bands[0]]
    for top, bottom in bands[1:]:
        prev_top, prev_bottom = out[-1]
        if top - prev_bottom <= gap:
            out[-1] = (prev_top, bottom)
        else:
            out.append((top, bottom))
    return out


def find_systems(
    image: Image.Image,
    *,
    min_ink_frac: float = _MIN_INK_FRAC,
    staff_row_frac: float = _STAFF_ROW_FRAC,
    merge_gap_frac: float = _MERGE_GAP_FRAC,
    margin_frac: float = _MARGIN_FRAC,
) -> list[SystemBox]:
    """Return ``(top, bottom)`` boxes, one per staff system on the page.

    Returns an empty list when no staff lines are found; callers then fall back
    to transcribing the whole page in one request.
    """
    gray = np.array(image.convert("L"))
    height, width = gray.shape
    if height < 10 or width < 10:
        return []
    ink = gray < _INK_THRESHOLD
    rows = ink.sum(axis=1)

    active = rows > max(1.0, min_ink_frac * width)
    gap = max(4, int(merge_gap_frac * height))
    bands = _merge(_runs(active), gap)

    # Keep only bands that actually contain a near-full-width horizontal line:
    # drops titles, lyrics and dynamics text while keeping real staff systems.
    staff_threshold = staff_row_frac * width
    kept = [b for b in bands if rows[b[0] : b[1]].max() >= staff_threshold]

    margin = int(margin_frac * height)
    return [(max(0, t - margin), min(height, b + margin)) for t, b in kept]


def _staff_bounds(ink_band: np.ndarray) -> tuple[int, int] | None:
    """Top/bottom row of the 5 staff lines inside a band, or ``None``."""
    _height, width = ink_band.shape
    rows = ink_band.sum(axis=1)
    line_rows = np.where(rows >= 0.5 * width)[0]
    if len(line_rows) < 5:
        return None
    return int(line_rows.min()), int(line_rows.max())


def count_barlines(image: Image.Image, systems: list[SystemBox]) -> int | None:
    """Count full-height vertical bar strokes inside the given systems.

    Returns ``None`` when the page has no recognisable staff systems, so the
    caller can skip the check instead of warning on garbage.
    """
    gray = np.array(image.convert("L"))
    ink = gray < _INK_THRESHOLD
    total = 0
    seen = False
    for top, bottom in systems:
        bounds = _staff_bounds(ink[top:bottom])
        if bounds is None:
            continue
        seen = True
        staff_top, staff_bottom = bounds
        band = ink[top + staff_top : top + staff_bottom + 1]
        staff_height = band.shape[0]
        if staff_height < 5:
            continue
        cols = band.sum(axis=0)
        # A barline touches every staff row; cluster thick/double strokes.
        runs = _runs(cols >= _BARLINE_FRAC * staff_height)
        total += len(runs)
    return total if seen else None


def estimate_measures(page_images: list[Image.Image], per_page_systems: list[list[SystemBox]]) -> int | None:
    """Estimated number of measures printed on the page(s).

    Barlines bound measures, so measures ≈ barlines (the final barline of each
    line is counted, multi-measure rests and repeats skew this slightly — the
    result is only used for a warning threshold, never as ground truth).
    """
    total = 0
    seen = False
    for page, systems in zip(page_images, per_page_systems):
        count = count_barlines(page, systems)
        if count is None:
            continue
        seen = True
        total += count
    return total if seen else None
