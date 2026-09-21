"""Wrapper around the MuseScore CLI for MusicXML → .mscz conversion."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CANDIDATES = ("musescore", "MuseScore", "mscore", "MuseScore4")

# Flatpak application IDs providing MuseScore 4 (checked after native binaries).
FLATPAK_APP_IDS = ("org.musescore.MuseScore",)


def find_musescore(explicit: str | Path | None = None) -> Path | None:
    """Locate a native MuseScore executable, or ``None`` if not installed.

    This only covers real binaries on ``PATH`` / well-known locations.
    For Flatpak installs see :func:`find_musescore_command`.
    """
    if explicit:
        p = Path(str(explicit).split()[0])
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


def _flatpak_command(app_id: str) -> list[str] | None:
    """Return ``[flatpak, run, app_id]`` if that Flatpak app is installed."""
    flatpak = shutil.which("flatpak")
    if flatpak is None:
        return None
    try:
        proc = subprocess.run(
            [flatpak, "info", app_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode == 0:
        return [flatpak, "run", app_id]
    return None


def find_musescore_command(explicit: str | Path | None = None) -> list[str] | None:
    """Locate MuseScore as an argv prefix, covering native + Flatpak installs.

    Returns e.g. ``["/usr/bin/musescore"]`` or
    ``["/usr/bin/flatpak", "run", "org.musescore.MuseScore"]``,
    or ``None`` when nothing is found.
    """
    if explicit:
        parts = str(explicit).split()
        if Path(parts[0]).exists():
            return parts
        return None
    native = find_musescore()
    if native is not None:
        return [str(native)]
    for app_id in FLATPAK_APP_IDS:
        cmd = _flatpak_command(app_id)
        if cmd is not None:
            return cmd
    return None


def is_available(explicit: str | Path | None = None) -> bool:
    return find_musescore_command(explicit) is not None


def _conversion_env() -> dict[str, str] | None:
    """Headless-safe env: force Qt offscreen when no display is present."""
    if os.environ.get("DISPLAY") or os.environ.get("QT_QPA_PLATFORM"):
        return None
    return {**os.environ, "QT_QPA_PLATFORM": "offscreen"}


def _outside_home(path: Path) -> bool:
    try:
        path.resolve().relative_to(Path.home().resolve())
        return False
    except ValueError:
        return True


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

    Flatpak installs only see ``$HOME`` (sandbox), so inputs outside
    the home directory are transparently staged via ``~/.cache``.
    """
    cmd = find_musescore_command(musescore_bin)
    if cmd is None:
        raise FileNotFoundError(
            "MuseScore executable not found (tried: "
            + ", ".join(CANDIDATES)
            + " and Flatpak "
            + ", ".join(FLATPAK_APP_IDS)
            + "). Install MuseScore 4 (`flatpak install flathub "
            + "org.musescore.MuseScore`) or keep --format musicxml."
        )
    output = output.with_suffix(".mscz")
    output.parent.mkdir(parents=True, exist_ok=True)

    is_flatpak = len(cmd) > 1
    if is_flatpak and (_outside_home(musicxml) or _outside_home(output.parent)):
        return _convert_via_staging(cmd, musicxml, output, timeout)

    proc = subprocess.run(
        [*cmd, str(musicxml), "-o", str(output)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=_conversion_env(),
    )
    if proc.returncode != 0 or not output.exists():
        raise RuntimeError(
            f"MuseScore conversion failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout)[-2000:]}"
        )
    return output


def _convert_via_staging(cmd: list[str], musicxml: Path, output: Path, timeout: int) -> Path:
    """Copy input under ``~/.cache`` (visible to Flatpak), convert, copy back."""
    staging_base = Path.home() / ".cache" / "pdf2mscz" / "flatpak-stage"
    staging_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mscz-", dir=staging_base) as tmp:
        staged_in = Path(tmp) / (musicxml.stem + ".musicxml")
        staged_out = Path(tmp) / (output.stem + ".mscz")
        shutil.copyfile(musicxml, staged_in)
        proc = subprocess.run(
            [*cmd, str(staged_in), "-o", str(staged_out)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_conversion_env(),
        )
        if proc.returncode != 0 or not staged_out.exists():
            raise RuntimeError(
                f"MuseScore conversion failed (exit {proc.returncode}): "
                f"{(proc.stderr or proc.stdout)[-2000:]}"
            )
        shutil.copyfile(staged_out, output)
    return output
