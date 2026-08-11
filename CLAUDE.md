# BT — Agent Coding Standards

These rules are non-negotiable for all code written in this repo. Follow them
without being reminded.

## File size

- **Max 600 lines per file, including blank lines and comments.**
- If a file is approaching the limit, split it before it hits the limit —
  don't wait until it's a problem. Split along responsibility, not
  arbitrarily (e.g. `bt_core/tools/system.py` + `bt_core/tools/apps.py`
  instead of one giant `tools.py`).
- One class or one cohesive group of related functions per file.

## Language & typing

- Python 3.11+ syntax: `match` statements, `X | Y` union types (not
  `Optional[X]`, not `Union`).
- Every function signature has full type hints — params and return type.
- Every data structure that crosses a boundary (config, tool args, LLM
  output, API responses) is a **Pydantic model**, not a raw dict.
- No `Any` unless there is genuinely no better type.

## Async

- Every I/O-bound call (network, disk, subprocess, audio stream) is
  `async def` and awaited. No blocking calls on the event loop.
- If a library is sync-only (e.g. some audio/TTS libs), wrap it with
  `asyncio.to_thread(...)` — never call it directly from an async context.

## Logging

- `structlog` only. **Never `print()`.**
- Every stage logs structured events with context (`stage`, `latency_ms`,
  relevant IDs) — not free-text strings.
- Errors are logged with enough context to debug without reproducing.

## Config

- No hardcoded values (paths, model names, timeouts, thresholds). Everything
  tunable lives in `config/config.yaml`, loaded via `pydantic-settings`.
- If you're about to hardcode a constant that isn't truly fixed (like a
  return status code), stop and add it to config instead.

## Docstrings & comments

- Google-style docstrings on every public function and class: one-line
  summary, `Args:`, `Returns:`, `Raises:` if relevant.
- Comments only explain *why*, never *what* — no comment should restate
  what the code already says via naming.

## Style

- Max line length: 100 (enforced by `ruff`, see `pyproject.toml`).
- No global mutable state. Pass dependencies in (constructor injection or
  function params) — don't reach for module-level singletons.
- Every package directory has `__init__.py`.

## Tools (`bt_core/tools/`)

Every tool BT can execute must have, without exception:

1. A Pydantic schema (name, args, description) used for LLM function-calling.
2. Async execution wrapped in a timeout.
3. `try/except` with a clear, spoken-friendly error message on failure —
   never let a raw traceback reach the user.
4. A permission tier: `safe` (auto-run), `confirm` (ask before running),
   or `admin` (requires explicit elevated opt-in). Default to the most
   restrictive tier that makes sense; don't default to `safe`.
5. Logged execution: tool name, args, result/error, latency.

## Error handling

- Fail-safe, not fail-silent: every pipeline stage (VAD, STT, LLM, TTS,
  tool exec) has a fallback behavior if it errors, and that fallback is
  logged.
- Don't add error handling for things that structurally can't happen
  (e.g. validating a Pydantic field that the type system already
  guarantees). Validate at real boundaries: mic input, LLM output parsing,
  external API responses, user-provided tool args.

## Testing

- New modules that have logic beyond wiring (parsing, validation, tool
  execution) get a test in `tests/`, using `pytest` + `pytest-asyncio`.
- Standalone-testable: every phase's deliverable should be runnable and
  verifiable on its own before it's wired into the full pipeline.

## Scope discipline

- Build exactly what the current phase asks for. Don't add abstractions,
  config options, or tools "for later" — add them when a later phase
  actually needs them.
- Don't refactor unrelated code while implementing a phase unless asked.
