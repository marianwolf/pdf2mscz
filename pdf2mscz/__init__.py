"""pdf2mscz — sheet music (PDF/PNG/JPG) → MusicXML → MuseScore (.mscz)."""

from pdf2mscz.pipeline import (
    ConversionOptions,
    ConversionOutput,
    TranscriptionError,
    convert,
)

__all__ = ["ConversionOptions", "ConversionOutput", "TranscriptionError", "convert"]
__version__ = "0.1.0"
