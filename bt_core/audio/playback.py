"""Speaker playback for BT.

sd.play/sd.wait block the calling thread, so playback runs via
asyncio.to_thread to keep the event loop free — unlike capture.py's
continuous callback stream, this is a one-off blocking call, which is
exactly what to_thread is for.
"""

from __future__ import annotations

import asyncio

import numpy as np
import sounddevice as sd

from bt_core.logging_setup import get_logger

log = get_logger(__name__)


async def play_audio(
    audio: np.ndarray,
    sample_rate: int,
    device: int | str | None = None,
) -> None:
    """Play a float32 audio buffer through the speakers and wait for it to finish.

    Args:
        audio: A ``(n_samples,)`` or ``(n_samples, channels)`` float32 array.
        sample_rate: Sample rate of ``audio``, in Hz.
        device: Output device index/name, or None for the system default.

    Raises:
        sounddevice.PortAudioError: If no usable output device is available.
    """
    log.info(
        "audio_playback_started",
        samples=len(audio),
        sample_rate=sample_rate,
        device=device,
    )
    try:
        await asyncio.to_thread(_play_blocking, audio, sample_rate, device)
    except Exception:
        log.error("audio_playback_failed", exc_info=True)
        raise
    log.info("audio_playback_finished")


def _play_blocking(audio: np.ndarray, sample_rate: int, device: int | str | None) -> None:
    """Blocking playback call, run off the event loop via asyncio.to_thread."""
    sd.play(audio, samplerate=sample_rate, device=device)
    sd.wait()
