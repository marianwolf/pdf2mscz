"""End-to-end pipeline: input → images → provider → MusicXML → .mscz.

Unlike the original pipeline this one refuses to write output that contains no
music: unusable model output raises :class:`TranscriptionError` (with the raw
model answer preserved on disk) instead of silently substituting a placeholder
score. VLM transcription can be split per staff system (`chunk="system"`),
which keeps each request small enough to fit the model's output-token budget.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from pdf2mscz.converters import get_provider_class
from pdf2mscz.converters.base import AbstractProvider, ProviderConfig
from pdf2mscz.utils.merge import first_attributes, merge_documents
from pdf2mscz.utils.musescore_cli import musicxml_to_mscz
from pdf2mscz.utils.pdf_utils import load_images, parse_pages, pdf_page_count
from pdf2mscz.utils.preprocess import preprocess_image
from pdf2mscz.utils.prompts import SYSTEM_PROMPT, fragment_prompt
from pdf2mscz.utils.staves import estimate_measures, find_systems
from pdf2mscz.utils.validate import ScoreStats, is_meaningful, validate_score
from pdf2mscz.utils.xml_sanitizer import SanitizeResult, as_document, sanitize_musicxml


class TranscriptionError(RuntimeError):
    """The provider produced no usable MusicXML (empty, prose or invalid)."""

    def __init__(
        self,
        reason: str,
        raw: str = "",
        raw_path: Path | None = None,
        detail: str = "",
    ) -> None:
        self.reason = reason
        self.raw = raw
        self.raw_path = raw_path
        self.detail = detail
        msg = f"transcription produced no usable MusicXML ({reason})"
        if detail:
            msg += f": {detail}"
        if raw_path:
            msg += f" [raw model output: {raw_path}]"
        super().__init__(msg)


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
    # --- WP1/WP2 knobs ---
    chunk: str = "system"  # system | page
    retry: int = 1  # extra repair attempts per request
    max_tokens: int = 8192  # backend output cap; 0 = let the server decide
    allow_empty: bool = False  # write the placeholder score anyway
    save_raw: bool = True  # keep the raw model answer next to the output
    on_progress: Callable[[str], None] | None = None


@dataclass
class ConversionOutput:
    musicxml_path: Path
    mscz_path: Path | None = None
    used_fallback: bool = False
    stats: ScoreStats | None = None
    warnings: list[str] = field(default_factory=list)
    raw_path: Path | None = None


def _progress(opts: ConversionOptions, message: str) -> None:
    if opts.on_progress:
        opts.on_progress(message)


def _build_provider(name: str, opts: ConversionOptions, *, model: str = "", api_key=None) -> AbstractProvider:
    cls = get_provider_class(name)
    key = api_key if api_key is not None else (opts.api_key if name == opts.provider else None)
    if key is None and cls.env_var:
        key = os.getenv(cls.env_var) or None
    config = ProviderConfig(
        model=model,
        api_key=key,
        # A custom base_url belongs to the main provider (e.g. a local
        # Ollama); forwarding it to a cloud refiner would send keys sideways.
        base_url=opts.base_url if name == opts.provider else None,
        temperature=opts.temperature,
        max_tokens=opts.max_tokens,
    )
    return cls(config)


def _complete_valid(
    provider: AbstractProvider,
    images: list[Image.Image],
    prompt: str,
    opts: ConversionOptions,
    raw_parts: list[str],
    warnings: list[str],
) -> tuple[str, SanitizeResult]:
    """One completion + as many repair rounds as the output needs."""
    text = provider.complete_text(prompt, images, opts.temperature)
    raw_parts.append(text)

    res = sanitize_musicxml(as_document(text))
    attempts = max(0, opts.retry)
    while (not res.ok or provider.last_truncated) and attempts > 0 and provider.supports_repair:
        attempts -= 1
        reason = res.reason or ("truncated output" if provider.last_truncated else "invalid")
        rep = provider.repair_musicxml(images, text, reason, res.detail)
        text = rep.musicxml
        raw_parts.append(text)
        res = sanitize_musicxml(as_document(text))

    if provider.last_truncated:
        msg = (
            f"model output hit the {opts.max_tokens or 'server'}-token limit; "
            "result may be incomplete (raise --max-tokens or use --chunk page)"
        )
        if msg not in warnings:
            warnings.append(msg)
    return text, res


def _transcribe(
    provider: AbstractProvider,
    images: list[Image.Image],
    opts: ConversionOptions,
    raw_parts: list[str],
    warnings: list[str],
) -> tuple[str, int | None]:
    """Return ``(candidate_musicxml, expected_measure_count_or_None)``."""
    use_chunk = opts.chunk == "system" and provider.supports_chunking

    if not use_chunk:
        _progress(opts, f"Transcribing {len(images)} page(s) in one request")
        text, _ = _complete_valid(provider, images, SYSTEM_PROMPT, opts, raw_parts, warnings)
        return text, None

    fragments: list[str] = []
    systems_by_page: list[list[tuple[int, int]]] = []
    whole_pages = 0
    context = ""
    for page_no, page in enumerate(images, start=1):
        systems = find_systems(page)
        systems_by_page.append(systems)
        if not systems:
            _progress(opts, f"Page {page_no}: no staff lines detected, sending whole page")
            text, _ = _complete_valid(provider, [page], SYSTEM_PROMPT, opts, raw_parts, warnings)
            fragments.append(text)
            whole_pages += 1
            continue
        _progress(opts, f"Page {page_no}/{len(images)}: {len(systems)} staff systems detected")
        for idx, (top, bottom) in enumerate(systems, start=1):
            crop = page.crop((0, top, page.width, bottom))
            prompt = fragment_prompt(context=context, first_line=(idx == 1 and page_no == 1))
            text, res = _complete_valid(provider, [crop], prompt, opts, raw_parts, warnings)
            if not res.ok:
                warnings.append(
                    f"page {page_no} system {idx}: unusable output ({res.reason}) — skipped"
                )
                continue
            if not context:
                context = first_attributes(text)
            fragments.append(text)
            stats = validate_score(as_document(text))
            _progress(
                opts,
                f"  system {idx}/{len(systems)}: {stats.measures} measures, "
                f"{stats.pitched_notes} notes",
            )

    expected = estimate_measures(images, systems_by_page) if any(systems_by_page) else None
    if not fragments:
        return "", expected
    if len(fragments) == 1 and whole_pages:
        # Single whole-page answer: keep the document as the model wrote it
        # (merging would flatten a multi-staff score into one part).
        return fragments[0], expected
    return merge_documents(fragments), expected


def _raw_dump(raw_parts: list[str]) -> str:
    if len(raw_parts) == 1:
        return raw_parts[0]
    return "\n\n".join(f"--- attempt {i} ---\n{part}" for i, part in enumerate(raw_parts, start=1))


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
    provider = _build_provider(opts.provider, opts, model=opts.model)
    raw_parts: list[str] = []
    warnings: list[str] = []
    candidate, expected = _transcribe(provider, images, opts, raw_parts, warnings)

    # 3. Optional AI refinement of a classical OMR result.
    if opts.refine_with:
        refiner = _build_provider(opts.refine_with, opts, model="", api_key=None)
        result = refiner.refine_musicxml(images, candidate)
        raw_parts.append(result.musicxml)
        candidate = result.musicxml
        if getattr(refiner, "last_truncated", False):
            warnings.append("refiner output hit its token limit; result may be incomplete")

    # 4. Validate BEFORE touching the filesystem: no output for empty scores.
    res = sanitize_musicxml(as_document(candidate), allow_empty=opts.allow_empty)
    stats = validate_score(res.xml) if (res.ok or res.placeholder) else ScoreStats()
    reason = res.reason if not res.ok else "no_pitched_notes"
    accepted = (res.ok and is_meaningful(stats)) or (opts.allow_empty and res.placeholder)

    if not accepted:
        raw_path = None
        if opts.save_raw and raw_parts:
            raw_path = out.with_suffix(".raw.txt")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(_raw_dump(raw_parts), encoding="utf-8")
        raise TranscriptionError(reason, _raw_dump(raw_parts), raw_path, res.detail)

    if res.recovered:
        warnings.append("output was truncated/damaged and has been repaired — verify it")
    if res.placeholder:
        warnings.append("placeholder score written (--allow-empty); nothing was transcribed")
    if expected and expected >= 4 and stats.measures < 0.5 * expected:
        warnings.append(
            f"only {stats.measures} of ~{expected} measures detected (barline estimate) — "
            "the transcription looks incomplete"
        )

    # 5. Write MusicXML (always; it's the interchange artifact).
    xml_path = out.with_suffix(".musicxml")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(res.xml, encoding="utf-8")

    # 6. MuseScore export unless music-only was requested.
    fmt = opts.output_format.lower()
    if fmt == "musicxml":
        return ConversionOutput(
            musicxml_path=xml_path, stats=stats, warnings=warnings
        )
    mscz_target = out if out.suffix.lower() == ".mscz" else out.with_suffix(".mscz")
    try:
        mscz_path = musicxml_to_mscz(xml_path, mscz_target, opts.musescore_bin)
        return ConversionOutput(
            musicxml_path=xml_path, mscz_path=mscz_path, stats=stats, warnings=warnings
        )
    except FileNotFoundError:
        if fmt == "both":
            return ConversionOutput(
                musicxml_path=xml_path,
                mscz_path=None,
                used_fallback=True,
                stats=stats,
                warnings=warnings,
            )
        raise
