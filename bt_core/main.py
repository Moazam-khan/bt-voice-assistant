"""BT — entrypoint.

Wires all components from config, loads models, and runs the assistant:
continuously listens on the microphone, and for each detected utterance,
runs it through the full pipeline (STT -> LLM/tools -> TTS) and speaks
the reply back.

Run with:
    python -m bt_core.main
"""

from __future__ import annotations

import asyncio

from bt_core.audio.capture import MicrophoneCapture
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


async def run() -> None:
    """Build BT's components and run the continuous listen-respond loop."""
    settings = get_settings()
    configure_logging(settings.logging)
    log.info("bt_starting", environment=settings.app.environment)

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

    log.info("bt_ready", wake_phrase=settings.wake_word.phrase)
    async with MicrophoneCapture(settings.audio) as mic:
        async for chunk in mic.stream():
            reply_audio = await pipeline.handle_chunk(chunk)
            if reply_audio is not None and len(reply_audio) > 0:
                await play_audio(reply_audio, settings.tts.sample_rate)


def main() -> None:
    """Synchronous entrypoint for `python -m bt_core.main`."""
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("bt_stopped_by_user")


if __name__ == "__main__":
    main()
