"""End-to-end pipeline test using synthesized speech as a mic stand-in.

Not part of bt_core — a throwaway verification script. Since no real
microphone is available in this dev environment, this generates real
spoken audio with Piper TTS (the wake phrase "hey jarvis" plus a command),
resamples it to the 16kHz the VAD/STT stage expects, pads it with silence
so VAD correctly fires speech-start/end the same way it would from a live
mic, and feeds it through the exact same Pipeline.handle_chunk() that
bt_core/main.py uses for live audio — wake word gating included, so a
command without "hey jarvis" first should produce no reply.

Run:
    python scripts/test_pipeline_e2e.py "what time is it"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bt_core.audio.playback import play_audio
from bt_core.audio.vad import VoiceActivityDetector
from bt_core.audio.wakeword import WakeWordDetector
from bt_core.config import get_settings
from bt_core.llm.client import OllamaClient
from bt_core.logging_setup import configure_logging, get_logger
from bt_core.pipeline import Pipeline
from bt_core.stt.transcriber import Transcriber
from bt_core.tools.registry import build_default_registry
from bt_core.tts.synthesizer import Synthesizer

log = get_logger(__name__)


def _resample_to_16k(audio: np.ndarray, source_rate: int) -> np.ndarray:
    """Resample audio to 16kHz, what the VAD/STT/wake-word stage requires."""
    resampled = resample_poly(audio, up=16000, down=source_rate)
    return resampled.astype(np.float32)


async def main() -> None:
    """Synthesize "hey jarvis" + a command, feed it through the pipeline, speak the reply."""
    command = " ".join(sys.argv[1:]) or "what time is it"

    settings = get_settings()
    configure_logging(settings.logging)
    system_prompt = (settings.paths.prompts_dir / "system.txt").read_text(encoding="utf-8")

    transcriber = Transcriber(settings.stt, settings.paths)
    synthesizer = Synthesizer(settings.tts)
    await asyncio.gather(transcriber.load(), synthesizer.load())

    pipeline = Pipeline(
        wake_word=WakeWordDetector(settings.wake_word),
        vad=VoiceActivityDetector(settings.vad, settings.audio.sample_rate),
        transcriber=transcriber,
        llm_client=OllamaClient(settings.llm),
        tool_registry=build_default_registry(settings),
        synthesizer=synthesizer,
        system_prompt=system_prompt,
        main_model=settings.llm.main_model,
    )

    print(f"Synthesizing: {settings.wake_word.phrase!r} + {command!r}")
    wake_audio = _resample_to_16k(
        await synthesizer.synthesize(settings.wake_word.phrase), settings.tts.sample_rate
    )
    command_audio = _resample_to_16k(
        await synthesizer.synthesize(command), settings.tts.sample_rate
    )

    silence = np.zeros(int(0.6 * settings.audio.sample_rate), dtype=np.float32)
    simulated_mic_audio = np.concatenate(
        [silence, wake_audio, silence, command_audio, silence, silence]
    )

    block_size = int(settings.audio.sample_rate * settings.audio.block_size_ms / 1000)
    reply_audio: np.ndarray | None = None
    for start in range(0, len(simulated_mic_audio), block_size):
        chunk = simulated_mic_audio[start : start + block_size]
        result = await pipeline.handle_chunk(chunk)
        if result is not None:
            reply_audio = result

    if reply_audio is None or len(reply_audio) == 0:
        print("No reply was produced (wake word or VAD never completed a turn).")
        return

    print("Playing BT's reply...")
    await play_audio(reply_audio, settings.tts.sample_rate)


if __name__ == "__main__":
    asyncio.run(main())
