"""Speech-to-text transcription for BT, using faster-whisper.

Model loading and inference are both CPU/GPU-bound blocking calls, so both
run via asyncio.to_thread to keep the event loop free. transcribe() returns
a lazy generator, so its iteration is done inside the same thread-pool call
as the transcribe() invocation — otherwise the real decoding work would
happen back on the event loop during iteration.

Model weights are cached under paths.models_dir/whisper, keeping them
inside the project's models/ folder instead of an implicit global cache.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import numpy as np

from bt_core.config import PathsConfig, SttConfig
from bt_core.logging_setup import get_logger

log = get_logger(__name__)

_FALLBACK_DEVICE = "cpu"
_FALLBACK_COMPUTE_TYPE = "int8"


def _register_cuda_dll_dirs() -> None:
    """Make pip-installed NVIDIA CUDA DLLs loadable by ctranslate2 on Windows.

    ctranslate2's CUDA backend (used by faster-whisper for GPU inference)
    loads cublas/cudnn/nvJitLink via a plain LoadLibrary call, which only
    honors the PATH environment variable — not os.add_dll_directory(), which
    only affects LoadLibraryEx callers that opt into that search mode (e.g.
    Python's own import machinery, ctypes.WinDLL). The nvidia-cublas-cu12 /
    nvidia-cudnn-cu12 / nvidia-nvjitlink-cu12 wheels place their DLLs under
    site-packages/nvidia/*/bin, which isn't on PATH by default. Without this,
    GPU transcription fails at inference time with "Library cublas64_12.dll
    is not found or cannot be loaded" even with the packages installed and
    the model loading successfully. Must run before any
    WhisperModel(device="cuda") is created.
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    bin_dirs = [
        str(bin_dir) for nvidia_root in nvidia.__path__ for bin_dir in Path(nvidia_root).glob("*/bin")
    ]
    if bin_dirs:
        os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ["PATH"]


_register_cuda_dll_dirs()

from faster_whisper import WhisperModel  # noqa: E402 — must follow DLL dir registration


class Transcriber:
    """Async wrapper around a faster-whisper model for speech-to-text."""

    def __init__(self, stt_config: SttConfig, paths_config: PathsConfig) -> None:
        """Initialize the transcriber. Model loading happens in :meth:`load`.

        Args:
            stt_config: STT section of BT's settings (model, device, compute
                type, language).
            paths_config: Paths section of BT's settings, used to locate the
                model weights cache directory.
        """
        self._config = stt_config
        self._download_root = paths_config.models_dir / "whisper"
        self._model: WhisperModel | None = None
        self._active_device: str | None = None

    async def load(self) -> None:
        """Load the whisper model, downloading weights on first run.

        Falls back to CPU/int8 if the configured device fails to load
        (e.g. GPU VRAM contention with Ollama), logging the fallback.
        """
        try:
            await self._load_on_device(self._config.device, self._config.compute_type)
        except Exception:
            log.error(
                "stt_model_load_failed_falling_back",
                model=self._config.model,
                requested_device=self._config.device,
                fallback_device=_FALLBACK_DEVICE,
                exc_info=True,
            )
            await self._load_on_device(_FALLBACK_DEVICE, _FALLBACK_COMPUTE_TYPE)

    async def _load_on_device(self, device: str, compute_type: str) -> None:
        """Construct the WhisperModel on a specific device.

        Args:
            device: ``"cuda"`` or ``"cpu"``.
            compute_type: e.g. ``"float16"`` (GPU) or ``"int8"`` (CPU).
        """
        log.info(
            "stt_model_loading", model=self._config.model, device=device, compute_type=compute_type
        )
        self._model = await asyncio.to_thread(
            WhisperModel,
            self._config.model,
            device=device,
            compute_type=compute_type,
            download_root=str(self._download_root),
        )
        self._active_device = device
        log.info("stt_model_loaded", model=self._config.model, device=device)

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a mono float32 audio buffer at 16kHz to text.

        If inference fails on the GPU (e.g. a missing/broken CUDA runtime
        library), falls back to CPU/int8 once and retries — logged either
        way — rather than crashing the pipeline.

        Args:
            audio: A ``(n_samples,)`` float32 array, 16kHz mono.

        Returns:
            The transcribed text, stripped of leading/trailing whitespace.
            Empty string if no speech was detected in the audio.

        Raises:
            RuntimeError: If called before :meth:`load`, or if transcription
                fails even after falling back to CPU.
        """
        if self._model is None:
            raise RuntimeError("Transcriber.load() must be called before transcribe()")

        try:
            text, detected_language, language_probability = await asyncio.to_thread(
                self._transcribe_blocking, audio
            )
        except Exception:
            if self._active_device == _FALLBACK_DEVICE:
                log.error("stt_transcription_failed", exc_info=True)
                raise
            log.error("stt_transcription_failed_falling_back_to_cpu", exc_info=True)
            await self._load_on_device(_FALLBACK_DEVICE, _FALLBACK_COMPUTE_TYPE)
            text, detected_language, language_probability = await asyncio.to_thread(
                self._transcribe_blocking, audio
            )

        log.info(
            "stt_transcribed",
            text_length=len(text),
            audio_duration_s=round(len(audio) / 16000, 2),
            detected_language=detected_language,
            language_probability=round(language_probability, 3),
        )
        return text

    def _transcribe_blocking(self, audio: np.ndarray) -> tuple[str, str, float]:
        """Run transcribe() and fully consume its lazy segment generator.

        Must only be called from a worker thread (via asyncio.to_thread) —
        this is where the actual blocking decode work happens.
        """
        assert self._model is not None
        segments, info = self._model.transcribe(audio, language=self._config.language)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, info.language, info.language_probability
