"""Microphone capture for BT.

Wraps sounddevice's callback-based InputStream in an asyncio-friendly
interface: audio chunks captured on PortAudio's internal callback thread
are pushed onto an asyncio.Queue and yielded to async consumers via
MicrophoneCapture.stream().
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import TracebackType

import numpy as np
import sounddevice as sd

from bt_core.config import AudioConfig
from bt_core.logging_setup import get_logger

log = get_logger(__name__)


class MicrophoneCapture:
    """Async wrapper around a sounddevice microphone input stream.

    Use as an async context manager; captured chunks are available via
    :meth:`stream` once the stream is open.

    Example:
        async with MicrophoneCapture(settings.audio) as mic:
            async for chunk in mic.stream():
                ...
    """

    def __init__(self, config: AudioConfig) -> None:
        """Initialize the capture wrapper.

        Args:
            config: Audio section of BT's settings (sample rate, channels,
                device, block size).
        """
        self._config = config
        self._block_size = int(config.sample_rate * config.block_size_ms / 1000)
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.InputStream | None = None

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """PortAudio callback — runs on sounddevice's internal audio thread."""
        if status:
            log.warning("audio_capture_status", status=str(status))
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, indata.copy())

    async def __aenter__(self) -> MicrophoneCapture:
        """Open and start the microphone stream."""
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            device=self._config.input_device,
            blocksize=self._block_size,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        log.info(
            "audio_capture_started",
            sample_rate=self._config.sample_rate,
            channels=self._config.channels,
            block_size=self._block_size,
            device=self._config.input_device,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop and close the microphone stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            log.info("audio_capture_stopped")

    async def stream(self) -> AsyncIterator[np.ndarray]:
        """Yield captured audio chunks as they arrive.

        Yields:
            One ``(block_size, channels)`` float32 numpy array per chunk.
        """
        while True:
            chunk = await self._queue.get()
            yield chunk
