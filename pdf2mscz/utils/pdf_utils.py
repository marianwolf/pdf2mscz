"""PDF page-range parsing + PDF/image loading."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_pages(spec: str | None, total: int) -> list[int]:
    """Parse ``--pages "1-3,5"`` (1-based) into 0-based indices.

    ``None``/``"all"`` → all pages. Out-of-range entries are dropped.
    """
    if spec is None or spec.strip().lower() in {"", "all"}:
        return list(range(total))
    wanted: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, _, b = chunk.partition("-")
            try:
                start, end = int(a) - 1, int(b) - 1
            except ValueError as exc:
                raise ValueError(f"Invalid page range {chunk!r}") from exc
            if end < start:
                start, end = end, start
            wanted.update(range(start, end + 1))
        else:
            try:
                wanted.add(int(chunk) - 1)
            except ValueError as exc:
                raise ValueError(f"Invalid page number {chunk!r}") from exc
    return sorted(i for i in wanted if 0 <= i < total)


def pdf_page_count(path: Path) -> int:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        return len(doc)
    finally:
        doc.close()


def pdf_to_images(path: Path, pages: list[int] | None = None, dpi: int = 300) -> list[Image.Image]:
    """Render PDF pages to PIL images at ``dpi``."""
    import pypdfium2 as pdfium

    scale = dpi / 72.0
    doc = pdfium.PdfDocument(str(path))
    try:
        idx = list(range(len(doc))) if pages is None else pages
        images: list[Image.Image] = []
        for i in idx:
            bitmap = doc[i].render(scale=scale).to_pil()
            images.append(bitmap.convert("RGB"))
        return images
    finally:
        doc.close()


def load_images(path: Path, pages: list[int] | None = None, dpi: int = 300) -> list[Image.Image]:
    """Load a PDF or a single image file as a list of PIL images."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        total = pdf_page_count(path)
        idx = list(range(total)) if pages is None else [i for i in pages if 0 <= i < total]
        return pdf_to_images(path, idx, dpi=dpi)
    if suffix in IMAGE_SUFFIXES:
        return [Image.open(path).convert("RGB")]
    raise ValueError(f"Unsupported input type: {suffix} (expected .pdf/.png/.jpg)")
