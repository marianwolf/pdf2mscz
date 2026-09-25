"""End-to-end pipeline: input → images → provider → MusicXML → .mscz.

Unlike the original pipeline this one refuses to write output that contains no
music: unusable model output raises :class:`TranscriptionError` (with the raw
model answer preserved on disk) instead of silently substituting a placeholder
score. VLM transcription can be split per staff system (`chunk="system"`),
which keeps each request small enough to fit the model's output-token budget.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
    jobs: int = 4  # parallel VLM requests (1 = fully serial)
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


def _preprocess_all(images: list[Image.Image], opts: ConversionOptions) -> list[Image.Image]:
    """Deskew + denoise every page, in parallel when --jobs allows it."""
    if opts.jobs > 1 and len(images) > 1:
        with ThreadPoolExecutor(max_workers=min(opts.jobs, len(images))) as pool:
            return list(pool.map(preprocess_image, images))
    return [preprocess_image(im) for im in images]


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
    # Hand back the *cleaned* document when it is usable: prose stripping and
    # measure de-duplication happen during sanitization.
    return (res.xml if res.ok else text), res


#: One provider request: a staff-system crop or a whole page.
@dataclass
class _Task:
    page_no: int
    images: list[Image.Image]
    is_fragment: bool  # False → whole page sent with the system prompt
    idx: int = 0  # system index on its page (1-based), 0 for whole pages
    n_systems: int = 0
    first_line: bool = False

    @property
    def label(self) -> str:
        if self.is_fragment:
            return f"page {self.page_no} system {self.idx}"
        return f"page {self.page_no}"


#: ``(text, raw attempts, per-task warnings, sanitize result)`` of one request.
_TaskResult = tuple[str, list[str], list[str], "SanitizeResult"]


def _plan_tasks(
    images: list[Image.Image], opts: ConversionOptions
) -> tuple[list[_Task], list[list[tuple[int, int]]]]:
    """Cut every page into its staff systems (serial; ~15 ms per page)."""
    tasks: list[_Task] = []
    systems_by_page: list[list[tuple[int, int]]] = []
    for page_no, page in enumerate(images, start=1):
        systems = find_systems(page)
        systems_by_page.append(systems)
        if not systems:
            _progress(opts, f"Page {page_no}: no staff lines detected, sending whole page")
            tasks.append(_Task(page_no=page_no, images=[page], is_fragment=False))
            continue
        _progress(opts, f"Page {page_no}/{len(images)}: {len(systems)} staff systems detected")
        for idx, (top, bottom) in enumerate(systems, start=1):
            tasks.append(
                _Task(
                    page_no=page_no,
                    # Crops are made here, up front: worker threads then only
                    # read their own image and never touch the shared page.
                    images=[page.crop((0, top, page.width, bottom))],
                    is_fragment=True,
                    idx=idx,
                    n_systems=len(systems),
                    first_line=(idx == 1 and page_no == 1),
                )
            )
    return tasks, systems_by_page


def _run_task(
    provider: AbstractProvider,
    task: _Task,
    context: str,
    opts: ConversionOptions,
) -> _TaskResult:
    """Run one request on a private copy of the provider.

    The copy matters: providers stash ``last_truncated`` on ``self``, so a
    shared instance would race when requests run in parallel.
    """
    prompt = (
        fragment_prompt(context=context, first_line=task.first_line)
        if task.is_fragment
        else SYSTEM_PROMPT
    )
    raw: list[str] = []
    warns: list[str] = []
    text, res = _complete_valid(copy.copy(provider), task.images, prompt, opts, raw, warns)
    return text, raw, warns, res


def _report_task(opts: ConversionOptions, task: _Task, result: _TaskResult) -> None:
    """Emit the per-system progress line (in reading order, main thread only)."""
    text, _raw, _warns, res = result
    if not (task.is_fragment and res.ok):
        return
    stats = validate_score(text)
    _progress(
        opts,
        f"  system {task.idx}/{task.n_systems}: {stats.measures} measures, "
        f"{stats.pitched_notes} notes",
    )


def _collect(
    tasks: list[_Task],
    results: list[_TaskResult | None],
    raw_parts: list[str],
    warnings: list[str],
) -> tuple[list[str], int]:
    """Merge per-task outputs into ordered fragments (reading order preserved)."""
    fragments: list[str] = []
    whole_pages = 0
    for task, result in zip(tasks, results):
        if result is None:  # pragma: no cover — every slot is filled before collect
            continue
        text, raw, warns, res = result
        raw_parts.extend(raw)
        warnings.extend(warns)
        if not task.is_fragment:
            fragments.append(text)
            whole_pages += 1
        elif not res.ok:
            warnings.append(f"{task.label}: unusable output ({res.reason}) — skipped")
        else:
            fragments.append(text)
    return fragments, whole_pages


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

    tasks, systems_by_page = _plan_tasks(images, opts)
    results: list[_TaskResult | None] = [None] * len(tasks)
    context = ""

    if opts.jobs <= 1 or len(tasks) == 1:
        # Serial: carried-over attributes may come from any successful
        # fragment, so later prompts keep improving as before.
        for i, task in enumerate(tasks):
            results[i] = result = _run_task(provider, task, context, opts)
            _report_task(opts, task, result)
            if task.is_fragment and result[3].ok and not context:
                context = first_attributes(result[0])
    else:
        # Parallel: the first request runs alone so the remaining prompts can
        # carry its established <attributes>; everything after that is issued
        # concurrently and collected in submission order (deterministic output).
        results[0] = first = _run_task(provider, tasks[0], "", opts)
        _report_task(opts, tasks[0], first)
        if tasks[0].is_fragment and first[3].ok:
            context = first_attributes(first[0])
        # If page 1 had no detectable systems, no context exists yet — the
        # rare case where parallel and serial prompts differ slightly.
        with ThreadPoolExecutor(max_workers=min(opts.jobs, len(tasks) - 1)) as pool:
            pending = [
                (i, pool.submit(_run_task, provider, task, context, opts))
                for i, task in enumerate(tasks[1:], start=1)
            ]
            try:
                for i, fut in pending:
                    results[i] = result = fut.result()
                    _report_task(opts, tasks[i], result)
            except BaseException:
                for _, fut in pending:
                    fut.cancel()  # don't fire off the remaining requests after an error
                raise

    fragments, whole_pages = _collect(tasks, results, raw_parts, warnings)

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
        images = load_images(src, pages=idx, dpi=opts.dpi, total=total)
    else:
        images = load_images(src, dpi=opts.dpi)
    if not images:
        raise ValueError("No pages selected (check --pages).")

    if opts.preprocess:
        images = _preprocess_all(images, opts)

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
