# BT — Personal Voice AI Assistant

BT is a local-first, 24/7 voice assistant for Windows. Wake word "Hey BT"
triggers a fully offline pipeline: speech capture → VAD → speech-to-text →
local LLM tool-calling (Ollama) → action execution → text-to-speech.

## Architecture

```
Mic → VAD → STT → LLM Router → Tool Executor → TTS → Speaker
                       ↓
                   Memory (ChromaDB + SQLite)
```

- **Async-first** — asyncio throughout, no blocking calls on the main loop.
- **Event-driven** — stages communicate over an internal message bus.
- **Config-driven** — all tunables live in `config/config.yaml`, nothing hardcoded.
- **Modular** — every stage (audio, STT, LLM, TTS, tools) is swappable independently.
- **Local-first** — no cloud calls unless explicitly opted into.
- **Observable** — structured logs (structlog) and per-stage latency metrics.

## Tech stack

| Concern       | Library                          |
|---------------|-----------------------------------|
| Audio I/O     | sounddevice, numpy                |
| VAD           | silero-vad                        |
| Wake word     | openWakeWord (custom "Hey BT")    |
| STT           | faster-whisper (distil-large-v3)  |
| LLM           | ollama (qwen2.5:14b / 7b / vl:7b) |
| Memory        | chromadb + sqlite                 |
| TTS           | piper-tts                         |
| Config        | pydantic-settings + PyYAML        |
| Logging       | structlog                         |
| Tool exec     | pyautogui, psutil, subprocess     |

## Project layout

```
bt_core/
├── audio/     # capture, VAD, playback
├── stt/       # whisper transcription
├── llm/       # ollama client, prompts, router
├── tts/       # piper synthesis
├── tools/     # executable actions (open_app, google_search, ...)
└── main.py    # orchestrator
config/
├── config.yaml
└── prompts/
logs/          # structured log output (gitignored)
models/        # whisper / piper / wakeword weights (gitignored, fetched separately)
```

## Development

Coding standards (file size limits, typing, async, logging, tool
requirements) are enforced via [`CLAUDE.md`](./CLAUDE.md). Read it before
contributing code.

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requires [Ollama](https://ollama.com) running locally with `qwen2.5:14b`,
`qwen2.5:7b`, `qwen2.5vl:7b`, and `nomic-embed-text` pulled.

## Status

Under active development — see commit history for phase progress
(config/logging → audio → STT → LLM tool-calling → tools → TTS →
full pipeline → wake word → memory → tray/auto-start → ERP → packaging).

## License

Private personal project.
