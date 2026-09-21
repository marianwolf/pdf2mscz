"""End-to-end pipeline: input → images → provider → MusicXML → .mscz."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pdf2mscz.converters import get_provider_class
from pdf2mscz.converters.base import ProviderConfig
from pdf2mscz.utils.musescore_cli import musicxml_to_mscz
from pdf2mscz.utils.pdf_utils import load_images, parse_pages, pdf_page_count
from pdf2mscz.utils.preprocess import preprocess_image
from pdf2mscz.utils.xml_sanitizer import sanitize_musicxml


@dataclass
class ConversionOptions:
    provider: str = "openai"
    model: str = ""
    api_key: str | None = None
    base_url: str | None = None
    pages: str | None = None
    dpi: int = 300
    preprocess: bool = False
    refine_with: str | None = None
    output_format: str = "mscz"  # mscz | musicxml | both
    musescore_bin: str | None = None
    temperature: float = 0.0


@dataclass
class ConversionOutput:
    musicxml_path: Path
    mscz_path: Path | None = None
    used_fallback: bool = False


def convert(
    input_path: str | Path,
    output: str | Path,
    options: ConversionOptions | None = None,
) -> ConversionOutput:
    """Run the full conversion. Raises on invalid input/provider errors."""
    opts = options or ConversionOptions()
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")
    out = Path(output)

    # 1. Load + select pages.
    if src.suffix.lower() == ".pdf":
        total = pdf_page_count(src)
        idx = parse_pages(opts.pages, total)
        images = load_images(src, pages=idx, dpi=opts.dpi)
    else:
        images = load_images(src, dpi=opts.dpi)
    if not images:
        raise ValueError("No pages selected (check --pages).")

    if opts.preprocess:
        images = [preprocess_image(im) for im in images]

    # 2. OMR / VLM transcription.
    provider_cls = get_provider_class(opts.provider)
    provider = provider_cls(
        ProviderConfig(
            model=opts.model,
            api_key=opts.api_key,
            base_url=opts.base_url,
            temperature=opts.temperature,
        )
    )
    result = provider.image_to_musicxml(images)

    # 3. Optional AI refinement of a classical OMR result.
    if opts.refine_with:
        refiner_cls = get_provider_class(opts.refine_with)
        refiner = refiner_cls(
            ProviderConfig(
                model="",
                api_key=opts.api_key,
                base_url=opts.base_url,
                temperature=0.0,
            )
        )
        result = refiner.refine_musicxml(images, result.musicxml)

    clean_xml = sanitize_musicxml(result.musicxml)

    # 4. Write MusicXML (always; it's the interchange artifact).
    if out.suffix.lower() in {".mscz", ".mscx"} or out.suffix.lower() in {".musicxml", ".xml"}:
        xml_path = out.with_suffix(".musicxml")
    else:
        xml_path = out.with_suffix(".musicxml")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(clean_xml, encoding="utf-8")

    # 5. MuseScore export unless musicxml-only was requested.
    fmt = opts.output_format.lower()
    if fmt == "musicxml":
        return ConversionOutput(musicxml_path=xml_path)
    mscz_target = out if out.suffix.lower() == ".mscz" else out.with_suffix(".mscz")
    try:
        mscz_path = musicxml_to_mscz(xml_path, mscz_target, opts.musescore_bin)
        return ConversionOutput(musicxml_path=xml_path, mscz_path=mscz_path)
    except FileNotFoundError:
        if fmt == "both":
            return ConversionOutput(musicxml_path=xml_path, mscz_path=None, used_fallback=True)
        raise
