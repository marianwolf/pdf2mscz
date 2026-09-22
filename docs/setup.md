# Setup

## Requirements

- Python 3.10+
- Optional: MuseScore 4 CLI (`musescore`/`mscore`) for `.mscz` export
- Optional: `oemer` for local classical OMR
- API keys for the VLM provider you use (see `.env.example`)

## Install

```bash
# PyPI (provider SDKs ship as extras — see README for the full list):
pip install "pdf2mscz[openai]"      # or [anthropic] [gemini] [nvidia] [ollama] [oemer] [all]

# from source:
pip install .
# with local OMR support:
pip install ".[oemer]"
# dev (tests, linters, build tooling):
pip install -e ".[dev]"
```

## API keys

```bash
cp .env.example .env
# edit .env, or export:
export OPENAI_API_KEY=sk-...
export NVIDIA_API_KEY=nvapi-...   # from build.nvidia.com
```

## Model choice

- **Vision/multimodal models only** — every request carries the page image.
  Text-only endpoints (e.g. NVIDIA "lightning" models such as
  `nvidia/nemotron-3.5-lightning-30b-a3b`) reject image parts with
  `400 … multimodal processing is not enabled`.
- Small VLMs transcribe more reliably when requests are split per staff system
  (`--chunk system`, the default) instead of sending the whole page at once.
- If answers come back truncated, raise `--max-tokens` (or keep
  `--chunk system`, whose answers are short).

## MuseScore CLI

- Ubuntu/Mint: `flatpak install flathub org.musescore.MuseScore` (empfohlen;
  das apt-Paket ist veraltet). pdf2mscz erkennt die Flatpak-Installation
  automatisch und ruft sie als `flatpak run org.musescore.MuseScore …` auf.
- macOS: `brew install --cask musescore`
- Verify: `pdf2mscz check-deps` (zeigt z. B. `/usr/bin/flatpak run org.musescore.MuseScore`)
- Alternativ explizit: `pdf2mscz convert … --musescore /usr/bin/musescore`
- Hinweis Flatpak-Sandbox: Die Flatpak-App sieht nur `$HOME`. Liegen Input/Output
  außerhalb (z. B. `/tmp`), staged pdf2mscz sie automatisch über `~/.cache/pdf2mscz/`.

Without MuseScore, use `--format musicxml` — the `.musicxml` can later be
opened and saved as `.mscz` in the MuseScore GUI.

## Local Ollama VLM

```bash
ollama pull llava:13b
ollama serve
pdf2mscz convert score.pdf out.mscz --provider ollama --format musicxml
```
