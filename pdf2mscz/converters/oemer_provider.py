"""Local classical OMR via the `oemer` package / CLI (no API key needed)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from pdf2mscz.converters.base import (
    AbstractProvider,
    ConversionResult,
    ProviderConfig,
    register_provider,
)


@register_provider
class OemerProvider(AbstractProvider):
    """Wraps `oemer` as a subprocess to avoid hard version pinning."""

    name = "oemer"
    requires_api_key = False

    def image_to_musicxml(self, images: list[Image.Image]) -> ConversionResult:
        exe = shutil.which("oemer")
        if exe is None:
            # Fall back to `python -m oemer` if only the library is installed.
            try:
                import oemer  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "oemer not found. Install with `pip install pdf2mscz[oemer]` "
                    "or install the oemer CLI."
                ) from exc
            return self._via_module(images)
        return self._via_cli(Path(exe), images)

    def _via_cli(self, exe: Path, images: list[Image.Image]) -> ConversionResult:
        with tempfile.TemporaryDirectory(prefix="pdf2mscz-oemer-") as tmp:
            tmpdir = Path(tmp)
            outs: list[str] = []
            for i, img in enumerate(images):
                src = tmpdir / f"page-{i + 1}.png"
                img.save(src)
                out_dir = tmpdir / f"out-{i + 1}"
                out_dir.mkdir()
                # `oemer <img> -o <out_dir>` writes MusicXML into out_dir.
                proc = subprocess.run(
                    [str(exe), str(src), "-o", str(out_dir)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"oemer failed: {proc.stderr[-2000:]}")
                xmls = sorted(out_dir.glob("*.musicxml")) + sorted(out_dir.glob("*.xml"))
                if not xmls:
                    raise RuntimeError(f"oemer produced no MusicXML in {out_dir}")
                outs.append(xmls[0].read_text(encoding="utf-8", errors="replace"))
            return ConversionResult(
                musicxml=_merge_parts(outs),
                confidence=0.6,
                provider=self.name,
                model="oemer",
            )

    def _via_module(self, images: list[Image.Image]) -> ConversionResult:
        # oemer's Python API varies across versions; shell out to the
        # module CLI for stability.
        with tempfile.TemporaryDirectory(prefix="pdf2mscz-oemer-") as tmp:
            tmpdir = Path(tmp)
            outs: list[str] = []
            for i, img in enumerate(images):
                src = tmpdir / f"page-{i + 1}.png"
                img.save(src)
                out_dir = tmpdir / f"out-{i + 1}"
                out_dir.mkdir()
                proc = subprocess.run(
                    ["python", "-m", "oemer", str(src), "-o", str(out_dir)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"oemer failed: {proc.stderr[-2000:]}")
                xmls = sorted(out_dir.glob("*.musicxml")) + sorted(out_dir.glob("*.xml"))
                if not xmls:
                    raise RuntimeError(f"oemer produced no MusicXML in {out_dir}")
                outs.append(xmls[0].read_text(encoding="utf-8", errors="replace"))
            return ConversionResult(
                musicxml=_merge_parts(outs),
                confidence=0.6,
                provider=self.name,
                model="oemer",
            )


def _merge_parts(docs: list[str]) -> str:
    """Naively return the first doc; multi-page merge is handled by pipeline."""
    return docs[0] if docs else ""


def build(config: ProviderConfig) -> OemerProvider:
    return OemerProvider(config)
