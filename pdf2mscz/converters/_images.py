"""Image <-> payload helpers shared by VLM providers."""

from __future__ import annotations

import base64
import io

from PIL import Image


def image_to_png_bytes(image: Image.Image, max_side: int = 2048) -> bytes:
    """Downscale (preserving aspect) and encode as PNG bytes."""
    img = image.convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def image_to_data_url(image: Image.Image) -> str:
    raw = image_to_png_bytes(image)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"
