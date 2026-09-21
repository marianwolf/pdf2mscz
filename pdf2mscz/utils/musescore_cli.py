"""Wrapper around the MuseScore CLI for MusicXML → .mscz conversion."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

CANDIDATES = ("musescore", "MuseScore", "mscore", "MuseScore4")


def find_musescore(explicit: str | Path | None = None) -> Path | None:
    """Locate the MuseScore executable, or ``None`` if not installed."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for name in CANDIDATES:
        found = shutil.which(name)
        if found:
            return Path(found)
    for p in (
        Path("/usr/bin/musescore"),
        Path("/usr/local/bin/musescore"),
        Path("/snap/bin/musescore"),
        Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore"),
    ):
        if p.exists():
            return p
    return None


def is_available(explicit: str | Path | None = None) -> bool:
    return find_musescore(explicit) is not None


def musicxml_to_mscz(
    musicxml: Path,
    output: Path,
    musescore_bin: str | Path | None = None,
    timeout: int = 300,
) -> Path:
    """Convert ``musicxml`` to ``.mscz`` headlessly.

    Returns the ``.mscz`` path on success. Raises
    :class:`FileNotFoundError` if MuseScore is missing and
    :class:`RuntimeError` if conversion fails.
    """
    exe = find_musescore(musescore_bin)
    if exe is None:
        raise FileNotFoundError(
            "MuseScore executable not found (tried: "
            + ", ".join(CANDIDATES)
            + "). Install MuseScore 4 or keep --format musicxml."
        )
    output = output.with_suffix(".mscz")
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(exe), str(musicxml), "-o", str(output)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0 or not output.exists():
        raise RuntimeError(
            f"MuseScore conversion failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout)[-2000:]}"
        )
    return output
