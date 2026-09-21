# Setup

## Requirements

- Python 3.10+
- Optional: MuseScore 4 CLI (`musescore`/`mscore`) for `.mscz` export
- Optional: `oemer` for local classical OMR
- API keys for the VLM provider you use (see `.env.example`)

## Install

```bash
pip install .
# with local OMR support:
pip install ".[oemer]"
# dev:
pip install -e ".[dev]"
```

## API keys

```bash
cp .env.example .env
# edit .env, or export:
export OPENAI_API_KEY=sk-...
```
3
## MuseScore CLI

- Ubuntu: `sudo snap install musescore` or apt build
- macOS: `brew install --cask musescore`
- Verify: `pdf2mscz check-deps`

Without MuseScore, use `--format musicxml` — the `.musicxml` can later be
opened and saved as `.mscz` in the MuseScore GUI.

## Local Ollama VLM

```bash
ollama pull llava:13b
ollama serve
pdf2mscz convert score.pdf out.mscz --provider ollama --format musicxml
```
