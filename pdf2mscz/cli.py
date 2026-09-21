"""Typer CLI: `pdf2mscz convert ...`."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from dotenv import load_dotenv

from pdf2mscz.converters import available_providers
from pdf2mscz.pipeline import ConversionOptions, convert
from pdf2mscz.utils.musescore_cli import find_musescore

load_dotenv()
app = typer.Typer(add_completion=False, help="PDF/PNG/JPG sheet music → MusicXML/.mscz")


def _resolve_key(provider: str, cli_key: str | None) -> str | None:
    if cli_key:
        return cli_key
    mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
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
) -> None:
    """Convert INPUT sheet music to OUTPUT (.mscz / MusicXML)."""
    providers = available_providers()
    if provider not in providers:
        typer.echo(f"Unknown provider {provider!r}. Available: {', '.join(providers)}", err=True)
        raise typer.Exit(2)
    if output_format not in {"mscz", "musicxml", "both"}:
        typer.echo("--format must be mscz|musicxml|both", err=True)
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
    )
    try:
        result = convert(input, output, opts)
    except FileNotFoundError as exc:
        typer.echo(f"{exc} Tip: install MuseScore 4 or use --format musicxml.", err=True)
        raise typer.Exit(1)
    except Exception as exc:  # keep CLI errors readable
        typer.echo(f"Conversion failed: {exc}", err=True)
        raise typer.Exit(1)
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
    exe = find_musescore()
    typer.echo(f"MuseScore: {exe if exe else 'NOT FOUND (only --format musicxml works)'}")
    typer.echo(f"Providers: {', '.join(available_providers())}")


# `pdf2mscz convert ...` is the canonical invocation; no default-callback
# so subcommands (`providers`, `check-deps`) dispatch cleanly.


def app_entry() -> None:
    app()


# `project.scripts` entry point.
def app_cli() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":
    app()
