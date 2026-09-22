"""Typer CLI: `pdf2mscz convert ...`."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from pdf2mscz.converters import available_providers
from pdf2mscz.pipeline import ConversionOptions, TranscriptionError, convert
from pdf2mscz.utils.musescore_cli import find_musescore_command

load_dotenv()
app = typer.Typer(add_completion=False, help="PDF/PNG/JPG sheet music → MusicXML/.mscz")


def _resolve_key(provider: str, cli_key: str | None) -> str | None:
    if cli_key:
        return cli_key
    mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "nvidia": "NVIDIA_API_KEY",
        "ollama": None,
        "oemer": None,
    }
    env = mapping.get(provider, None)
    if env is None:
        return None
    names = [env] if isinstance(env, str) else env
    for n in names:
        if os.getenv(n):
            return os.getenv(n)
    return None


def _oemer_available() -> bool:
    return shutil.which("oemer") is not None or importlib.util.find_spec("oemer") is not None


def _report_failure(exc: TranscriptionError) -> None:
    typer.echo(f"Transcription failed: no usable MusicXML ({exc.reason})", err=True)
    if exc.detail:
        typer.echo(f"  parser: {exc.detail}", err=True)
    if exc.raw_path:
        typer.echo(f"  raw model output saved to {exc.raw_path}", err=True)
    else:
        tail = (exc.raw or "").strip()
        if tail:
            typer.echo("  raw model output (tail):", err=True)
            typer.echo(tail[-1500:], err=True)


def _should_retry_oemer(force: bool) -> bool:
    """Ask (TTY) or honour --with-oemer before falling back to classical OMR."""
    if not _oemer_available():
        typer.echo(
            "Tip: `pip install pdf2mscz[oemer]` adds a local OMR fallback (--with-oemer).",
            err=True,
        )
        return False
    if force:
        return True
    if not sys.stdin.isatty():
        typer.echo("Hint: add --with-oemer to retry with the local OMR engine.", err=True)
        return False
    try:
        return typer.confirm("Retry with the local OMR engine (oemer)?", default=True)
    except (typer.Abort, KeyboardInterrupt, EOFError):
        return False


@app.command(name="convert")
def convert_cmd(
    input: Path = typer.Argument(..., exists=True, help="Input PDF or image (PNG/JPG)."),
    output: Path = typer.Argument(..., help="Output file (.mscz or .musicxml)."),
    provider: str = typer.Option("openai", "--provider", "-p", help="OMR/VLM provider."),
    model: str = typer.Option(
        "", "--model", "-m", help="Model override (provider default if empty)."
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="API key (else from .env)."),
    base_url: str | None = typer.Option(
        None, "--base-url", help="Custom endpoint (Ollama/OpenAI-compatible)."
    ),
    pages: str | None = typer.Option(None, "--pages", help='Page range, e.g. "1-3,5" or "all".'),
    dpi: int = typer.Option(300, "--dpi", min=72, max=600),
    preprocess: bool = typer.Option(
        False, "--preprocess/--no-preprocess", help="Deskew + denoise."
    ),
    refine_with: str | None = typer.Option(
        None, "--refine", help="AI provider refining OMR output."
    ),
    output_format: str = typer.Option("mscz", "--format", "-f", help="mscz | musicxml | both."),
    musescore_bin: str | None = typer.Option(
        None, "--musescore", help="Explicit MuseScore binary."
    ),
    temperature: float = typer.Option(0.0, "--temperature"),
    chunk: str = typer.Option(
        "system",
        "--chunk",
        help="VLM request granularity: system (one call per staff line) | page.",
    ),
    retry: int = typer.Option(
        1, "--retry", min=0, max=5, help="Repair attempts per request when output is invalid."
    ),
    max_tokens: int = typer.Option(
        8192, "--max-tokens", min=0, help="Backend output cap; 0 = server default."
    ),
    allow_empty: bool = typer.Option(
        False, "--allow-empty", help="Write a placeholder score instead of failing."
    ),
    save_raw: bool = typer.Option(
        True, "--save-raw/--no-save-raw", help="Keep the raw model answer as OUTPUT.raw.txt."
    ),
    with_oemer: bool = typer.Option(
        False, "--with-oemer", help="On failure, retry with the local oemer OMR engine."
    ),
) -> None:
    """Convert INPUT sheet music to OUTPUT (.mscz / MusicXML)."""
    providers = available_providers()
    if provider not in providers:
        typer.echo(f"Unknown provider {provider!r}. Available: {', '.join(providers)}", err=True)
        raise typer.Exit(2)
    if output_format not in {"mscz", "musicxml", "both"}:
        typer.echo("--format must be mscz|musicxml|both", err=True)
        raise typer.Exit(2)
    if chunk not in {"system", "page"}:
        typer.echo("--chunk must be system|page", err=True)
        raise typer.Exit(2)

    opts = ConversionOptions(
        provider=provider,
        model=model,
        api_key=_resolve_key(provider, api_key),
        base_url=base_url,
        pages=pages,
        dpi=dpi,
        preprocess=preprocess,
        refine_with=refine_with,
        output_format=output_format,
        musescore_bin=musescore_bin,
        temperature=temperature,
        chunk=chunk,
        retry=retry,
        max_tokens=max_tokens,
        allow_empty=allow_empty,
        save_raw=save_raw,
    )
    opts.on_progress = lambda msg: typer.echo(msg)

    try:
        result = convert(input, output, opts)
    except TranscriptionError as exc:
        _report_failure(exc)
        if provider == "oemer" or not _should_retry_oemer(with_oemer):
            raise typer.Exit(1)
        typer.echo("Retrying with local OMR (oemer)…", err=True)
        opts.provider = "oemer"
        opts.model = ""
        opts.chunk = "page"
        opts.api_key = None
        try:
            result = convert(input, output, opts)
        except TranscriptionError as retry_exc:
            _report_failure(retry_exc)
            raise typer.Exit(1)
        except Exception as retry_exc:  # keep CLI errors readable
            typer.echo(f"Conversion failed: {retry_exc}", err=True)
            raise typer.Exit(1)
    except FileNotFoundError as exc:
        typer.echo(f"{exc} Tip: install MuseScore 4 or use --format musicxml.", err=True)
        raise typer.Exit(1)
    except Exception as exc:  # keep CLI errors readable
        typer.echo(f"Conversion failed: {exc}", err=True)
        raise typer.Exit(1)

    for warning in result.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    if result.stats:
        typer.echo(f"Transcribed {result.stats.summary()}")
    typer.echo(f"Wrote MusicXML → {result.musicxml_path}")
    if result.mscz_path:
        typer.echo(f"Wrote MuseScore → {result.mscz_path}")
    elif result.used_fallback:
        typer.echo("MuseScore not found — kept MusicXML only.", err=True)


@app.command(name="providers")
def list_providers() -> None:
    for name in available_providers():
        typer.echo(name)


@app.command(name="check-deps")
def check_deps() -> None:
    cmd = find_musescore_command()
    typer.echo(f"MuseScore: {' '.join(cmd) if cmd else 'NOT FOUND (only --format musicxml works)'}")
    typer.echo(f"Providers: {', '.join(available_providers())}")
    typer.echo(f"oemer fallback: {'available' if _oemer_available() else 'not installed'}")


# `pdf2mscz convert ...` is the canonical invocation; no default-callback
# so subcommands (`providers`, `check-deps`) dispatch cleanly.


def app_entry() -> None:
    app()


# `project.scripts` entry point.
def app_cli() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":
    app()
