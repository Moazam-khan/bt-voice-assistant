"""Typed configuration loader for BT.

Loads and validates ``config/config.yaml`` into Pydantic models. Every other
module reads settings through :func:`get_settings` — nothing in the codebase
should read the YAML file directly or hardcode a tunable value.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """Find BT's root directory: where config/ and models/ live.

    When run from source, that's two levels up from this file
    (bt_core/config.py -> project root). When packaged by PyInstaller
    (sys.frozen is set), __file__ points inside a temp extraction dir
    instead, so config/models must live next to the .exe and are found
    via sys.executable's directory instead.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _expand_path(value: str | Path) -> Path:
    """Expand env vars/``~`` in a path, and resolve relative paths against
    BT's root directory so config.yaml doesn't need machine-specific
    absolute paths (making the app relocatable, e.g. once packaged).

    Args:
        value: A raw path, possibly containing unexpanded env vars, and
            possibly relative (e.g. "models/whisper").

    Returns:
        The fully expanded, absolute ``Path``.
    """
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    if not expanded.is_absolute():
        expanded = _project_root() / expanded
    return expanded


ExpandedPath = Annotated[Path, BeforeValidator(_expand_path)]


class AppConfig(BaseModel):
    """Top-level application identity."""

    name: str
    environment: Literal["dev", "prod"]


class PathsConfig(BaseModel):
    """Filesystem locations used across the project."""

    root: ExpandedPath
    models_dir: ExpandedPath
    logs_dir: ExpandedPath
    prompts_dir: ExpandedPath


class LoggingConfig(BaseModel):
    """structlog output settings."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    format: Literal["pretty", "json"]
    file: ExpandedPath


class AudioConfig(BaseModel):
    """Microphone capture and speaker playback settings."""

    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    input_device: int | str | None = None
    output_device: int | str | None = None
    block_size_ms: int = Field(gt=0)


class VadConfig(BaseModel):
    """Voice activity detection (silero-vad) settings."""

    model: str
    threshold: float = Field(ge=0.0, le=1.0)
    min_speech_ms: int = Field(gt=0)
    min_silence_ms: int = Field(gt=0)


class WakeWordConfig(BaseModel):
    """Wake word ("Hey BT") detection settings."""

    phrase: str
    model_path: ExpandedPath
    threshold: float = Field(ge=0.0, le=1.0)


class SttConfig(BaseModel):
    """Speech-to-text (faster-whisper) settings."""

    model: str
    device: Literal["cuda", "cpu"]
    compute_type: str
    language: str


class LlmConfig(BaseModel):
    """Ollama connection and model selection."""

    host: str
    main_model: str
    fast_model: str
    vision_model: str
    embed_model: str
    timeout_s: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=2.0)
    history_turns: int = Field(gt=0)


class TtsConfig(BaseModel):
    """Text-to-speech (Piper) settings."""

    engine: str
    voice_model_path: ExpandedPath
    sample_rate: int = Field(gt=0)


class MemoryConfig(BaseModel):
    """Structured (SQLite) and vector (ChromaDB) memory locations."""

    chroma_path: ExpandedPath
    sqlite_path: ExpandedPath


class WeatherConfig(BaseModel):
    """Sidebar weather widget settings.

    This is BT's only feature that requires internet access — everything
    else runs locally.
    """

    default_city: str
    refresh_minutes: int = Field(gt=0)


class ToolsConfig(BaseModel):
    """Defaults applied to every executable tool."""

    screenshot_dir: ExpandedPath
    default_permission_tier: Literal["safe", "confirm", "admin"]
    timeout_s: int = Field(gt=0)


class LatencyTargetsMsConfig(BaseModel):
    """Per-stage latency budgets, in milliseconds, used for metrics/alerts."""

    wake_word_detection: int = Field(gt=0)
    first_spoken_word: int = Field(gt=0)
    simple_command_total: int = Field(gt=0)


class BTSettings(BaseSettings):
    """Root settings object — the full validated contents of config.yaml.

    Values are loaded from ``config/config.yaml`` via :meth:`from_yaml`.
    Individual fields can be overridden with environment variables prefixed
    ``BT_`` using ``__`` as the nested delimiter, e.g. ``BT_LLM__HOST``.
    """

    model_config = SettingsConfigDict(
        env_prefix="BT_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    app: AppConfig
    paths: PathsConfig
    logging: LoggingConfig
    audio: AudioConfig
    vad: VadConfig
    wake_word: WakeWordConfig
    stt: SttConfig
    llm: LlmConfig
    tts: TtsConfig
    memory: MemoryConfig
    weather: WeatherConfig
    tools: ToolsConfig
    apps: dict[str, ExpandedPath]
    latency_targets_ms: LatencyTargetsMsConfig

    @classmethod
    def from_yaml(cls, path: Path) -> BTSettings:
        """Build settings from a YAML config file.

        Args:
            path: Path to a YAML file matching BT's config schema.

        Returns:
            A fully validated ``BTSettings`` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            pydantic.ValidationError: If the YAML contents fail validation.
        """
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)


_DEFAULT_CONFIG_PATH = _project_root() / "config" / "config.yaml"


@lru_cache(maxsize=1)
def get_settings(config_path: Path = _DEFAULT_CONFIG_PATH) -> BTSettings:
    """Load and cache BT's settings.

    Cached via ``lru_cache`` so the YAML file is only parsed once per process.

    Args:
        config_path: Path to the YAML config file. Defaults to
            ``config/config.yaml`` at the project root.

    Returns:
        The cached, validated ``BTSettings`` instance.
    """
    return BTSettings.from_yaml(config_path)
