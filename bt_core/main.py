"""BT — entrypoint.

Wires all components from config, loads models, and runs the assistant:
continuously listens on the microphone, and for each detected utterance,
runs it through the full pipeline (STT -> LLM/tools -> TTS) and speaks
the reply back, while a chat window shows the transcript.

pywebview's chat window must run on the process's main thread (it owns
the native GUI event loop), so the async pipeline runs on a dedicated
background thread instead, started once the window is ready.

Run with:
    python -m bt_core.main
"""

from __future__ import annotations

import asyncio
import os
import threading

from bt_core.audio.capture import MicrophoneCapture
from bt_core.audio.playback import play_audio
from bt_core.audio.vad import VoiceActivityDetector
from bt_core.audio.wakeword import WakeWordDetector
from bt_core.config import BTSettings, get_settings
from bt_core.llm.client import OllamaClient
from bt_core.logging_setup import configure_logging, get_logger
from bt_core.memory.structured import ConversationStore
from bt_core.memory.vector import SemanticMemory
from bt_core.pipeline import Pipeline
from bt_core.stt.transcriber import Transcriber
from bt_core.system_stats import get_system_stats
from bt_core.tools.base import ToolError
from bt_core.tools.registry import build_default_registry
from bt_core.tools.weather import fetch_weather
from bt_core.tray import TrayIcon
from bt_core.tts.synthesizer import Synthesizer
from bt_core.ui.chat_window import ChatWindow

log = get_logger(__name__)

_SYSTEM_STATS_REFRESH_S = 2


async def _listen_loop(pipeline: Pipeline, settings: BTSettings, quit_event: asyncio.Event) -> None:
    """Consume the microphone stream until quit_event is set.

    Args:
        pipeline: The built conversation pipeline.
        settings: BT's full settings.
        quit_event: Set (from the tray icon) to stop listening.
    """
    async with MicrophoneCapture(settings.audio) as mic:
        async for chunk in mic.stream():
            if quit_event.is_set():
                return
            reply_audio = await pipeline.handle_chunk(chunk)
            if reply_audio is not None and len(reply_audio) > 0:
                await play_audio(reply_audio, settings.tts.sample_rate)


async def _handle_text_and_play(pipeline: Pipeline, settings: BTSettings, text: str) -> None:
    """Run one text-chat turn and play back BT's spoken reply.

    Args:
        pipeline: The built conversation pipeline.
        settings: BT's full settings.
        text: The user's typed message.
    """
    reply_audio = await pipeline.handle_text(text)
    if reply_audio is not None and len(reply_audio) > 0:
        await play_audio(reply_audio, settings.tts.sample_rate)


async def _weather_refresh_loop(chat_window: ChatWindow, settings: BTSettings) -> None:
    """Fetch and display weather on startup, then refresh periodically.

    BT's only network-dependent feature — a failure here (no internet,
    city not found) just shows an "unavailable" state, logged, and never
    affects the rest of the app.

    Args:
        chat_window: The chat window to push weather updates into.
        settings: BT's full settings.
    """
    while True:
        try:
            report = await fetch_weather(settings.weather.default_city)
            chat_window.set_weather(report.city, report.temperature_c, report.description)
        except ToolError:
            log.warning(
                "weather_refresh_failed", city=settings.weather.default_city, exc_info=True
            )
            chat_window.set_weather_unavailable()
        await asyncio.sleep(settings.weather.refresh_minutes * 60)


async def _system_stats_loop(chat_window: ChatWindow) -> None:
    """Push live CPU/RAM/disk usage into the sidebar every few seconds.

    Fully local (psutil reads the OS directly) — unlike weather, this
    has no failure mode worth handling specially.

    Args:
        chat_window: The chat window to push stats updates into.
    """
    while True:
        stats = await get_system_stats()
        chat_window.set_system_stats(
            stats.cpu_percent,
            stats.ram_percent,
            stats.ram_used_gb,
            stats.ram_total_gb,
            stats.disk_percent,
            stats.disk_used_gb,
            stats.disk_total_gb,
        )
        await asyncio.sleep(_SYSTEM_STATS_REFRESH_S)


async def _async_main(chat_window: ChatWindow) -> None:
    """Build BT's async components and run the continuous listen-respond loop.

    Args:
        chat_window: The already-created chat window, to wire the
            pipeline's status/message events into.
    """
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
        conversation_store=ConversationStore(settings.memory.sqlite_path),
        semantic_memory=SemanticMemory(
            settings.memory.chroma_path, settings.llm.embed_model, settings.llm.host
        ),
        system_prompt=system_prompt,
        main_model=settings.llm.main_model,
        on_status_change=chat_window.set_status,
        on_user_text=chat_window.show_user_message,
        on_assistant_text=chat_window.show_bt_message,
        on_tool_used=chat_window.show_tool_action,
        on_confirmation_needed=chat_window.show_confirmation_prompt,
    )

    loop = asyncio.get_running_loop()
    quit_event = asyncio.Event()
    tray = TrayIcon(on_quit=lambda: loop.call_soon_threadsafe(quit_event.set))
    tray.start()
    chat_window.set_start_listening_handler(
        lambda: loop.call_soon_threadsafe(pipeline.trigger_listening)
    )
    chat_window.set_text_message_handler(
        lambda text: asyncio.run_coroutine_threadsafe(
            _handle_text_and_play(pipeline, settings, text), loop
        )
    )
    chat_window.set_open_config_handler(lambda: os.startfile(str(settings.paths.root / "config")))
    chat_window.set_confirmation_response_handler(
        lambda allowed: loop.call_soon_threadsafe(pipeline.respond_confirmation, allowed)
    )

    chat_window.set_session_info(settings.wake_word.phrase, settings.llm.main_model)
    chat_window.set_status("idle")
    weather_task = asyncio.create_task(_weather_refresh_loop(chat_window, settings))
    system_stats_task = asyncio.create_task(_system_stats_loop(chat_window))
    log.info("bt_ready", wake_phrase=settings.wake_word.phrase)
    listen_task = asyncio.create_task(_listen_loop(pipeline, settings, quit_event))
    quit_task = asyncio.create_task(quit_event.wait())
    await asyncio.wait([listen_task, quit_task], return_when=asyncio.FIRST_COMPLETED)

    if listen_task.done() and not listen_task.cancelled() and listen_task.exception() is not None:
        log.error("bt_listen_loop_crashed", exc_info=listen_task.exception())
        chat_window.set_status("idle")
        chat_window.show_error_message(
            "I can't hear you right now — something's wrong with my microphone. "
            "You can still type to me below while that gets fixed."
        )
        # Voice is down, but text chat still works — stay alive until Quit
        # instead of shutting the whole backend down over a mic failure.
        if not quit_task.done():
            await quit_task
    else:
        listen_task.cancel()

    weather_task.cancel()
    system_stats_task.cancel()
    tray.stop()
    log.info("bt_shutting_down")


def main() -> None:
    """Synchronous entrypoint for `python -m bt_core.main`.

    The chat window owns the main thread; the async pipeline runs on a
    background thread, started once the window is ready to receive
    updates.
    """
    chat_window = ChatWindow()

    def start_backend() -> None:
        asyncio.run(_async_main(chat_window))

    backend_thread = threading.Thread(target=start_backend, daemon=True)
    try:
        chat_window.start(on_ready=backend_thread.start)
    except KeyboardInterrupt:
        log.info("bt_stopped_by_user")


if __name__ == "__main__":
    main()
