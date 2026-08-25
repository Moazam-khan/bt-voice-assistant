"""Chat window UI for BT, using pywebview.

pywebview's start() blocks its calling thread with the OS's native GUI
event loop, so it must run on the process's main thread — a stricter
requirement than the tray icon's background-thread pattern. The async
pipeline instead runs on a separate background thread; all Python->JS
calls happen via Window.evaluate_js, which pywebview documents as safe
to call from any thread once the window exists.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import webview

from bt_core.logging_setup import get_logger

log = get_logger(__name__)

_HTML_PATH = Path(__file__).resolve().parent / "chat.html"
_ICON_PATH = Path(__file__).resolve().parent / "icon.ico"


class _ChatApi:
    """Exposed to JS as ``window.pywebview.api`` — one method per UI action.

    A separate object (rather than methods directly on ChatWindow) because
    pywebview publishes every public method of the js_api object to JS
    automatically; keeping it minimal avoids accidentally exposing
    ChatWindow's other methods (show_user_message, etc.) to the page.
    """

    def __init__(self, chat_window: ChatWindow) -> None:
        self._chat_window = chat_window

    def start_listening(self) -> None:
        """Called from JS when the user clicks the "Start" button."""
        self._chat_window._on_start_listening()

    def send_text_message(self, text: str) -> None:
        """Called from JS when the user submits the text input box."""
        self._chat_window._on_text_message(text)

    def open_config_folder(self) -> None:
        """Called from JS when the user clicks the settings gear icon."""
        self._chat_window._on_open_config()


class ChatWindow:
    """Wraps a pywebview window showing BT's conversation transcript."""

    def __init__(self) -> None:
        """Create the window (not shown until :meth:`start` is called)."""
        self._on_start_listening: Callable[[], None] = lambda: None
        self._on_text_message: Callable[[str], None] = lambda text: None
        self._on_open_config: Callable[[], None] = lambda: None
        html = _HTML_PATH.read_text(encoding="utf-8")
        self._window = webview.create_window(
            "BT",
            html=html,
            width=920,
            height=600,
            min_size=(700, 480),
            background_color="#0f1115",
            js_api=_ChatApi(self),
        )

    def set_start_listening_handler(self, handler: Callable[[], None]) -> None:
        """Set the callback invoked when the user clicks the "Start" button.

        Args:
            handler: Called with no arguments, on pywebview's own calling
                thread — if it needs to reach the asyncio pipeline (on a
                different thread), it must schedule that itself (e.g. via
                ``loop.call_soon_threadsafe``).
        """
        self._on_start_listening = handler

    def set_text_message_handler(self, handler: Callable[[str], None]) -> None:
        """Set the callback invoked when the user submits typed text.

        Args:
            handler: Called with the typed message, on pywebview's own
                calling thread — if it needs to reach the asyncio pipeline
                (on a different thread), it must schedule that itself.
        """
        self._on_text_message = handler

    def set_open_config_handler(self, handler: Callable[[], None]) -> None:
        """Set the callback invoked when the user clicks the settings gear icon.

        Args:
            handler: Called with no arguments, on pywebview's own calling
                thread.
        """
        self._on_open_config = handler

    def start(self, on_ready: Callable[[], None]) -> None:
        """Show the window and block the calling thread until it's closed.

        Must be called from the main thread.

        Args:
            on_ready: Called once the window is created and ready to
                receive JS calls — the natural place to start background
                work (e.g. the pipeline's asyncio loop on another thread).
        """
        webview.start(on_ready, icon=str(_ICON_PATH))

    def show_user_message(self, text: str) -> None:
        """Append a user message bubble to the transcript."""
        self._call_js("addUserMessage", text)

    def show_bt_message(self, text: str) -> None:
        """Append a BT message bubble to the transcript."""
        self._call_js("addBtMessage", text)

    def show_error_message(self, text: str) -> None:
        """Append a visually distinct error notice (e.g. a mic failure)."""
        self._call_js("addErrorMessage", text)

    def set_status(self, status: str) -> None:
        """Update the status indicator.

        Args:
            status: One of "idle", "listening", "thinking", "speaking".
        """
        self._call_js("setStatus", status)

    def set_session_info(self, wake_phrase: str, model_name: str) -> None:
        """Populate the sidebar's static session details, once at startup.

        Args:
            wake_phrase: The configured wake word phrase.
            model_name: The main LLM model name in use.
        """
        self._call_js("setSessionInfo", wake_phrase, model_name)

    def set_weather(self, city: str, temperature_c: float, description: str) -> None:
        """Update the sidebar's weather card.

        Args:
            city: Resolved city name.
            temperature_c: Current temperature in Celsius.
            description: Short conditions description (e.g. "overcast").
        """
        self._call_js("setWeather", city, f"{temperature_c:.0f}°C", description)

    def set_weather_unavailable(self) -> None:
        """Show a plain "unavailable" state in the weather card (e.g. no internet)."""
        self._call_js("setWeatherUnavailable")

    def set_system_stats(
        self,
        cpu_percent: float,
        ram_percent: float,
        ram_used_gb: float,
        ram_total_gb: float,
        disk_percent: float,
        disk_used_gb: float,
        disk_total_gb: float,
    ) -> None:
        """Update the sidebar's live CPU/RAM/disk usage card.

        Args:
            cpu_percent: Current CPU usage, 0-100.
            ram_percent: Current RAM usage, 0-100.
            ram_used_gb: RAM currently in use, in GB.
            ram_total_gb: Total installed RAM, in GB.
            disk_percent: Current disk usage, 0-100.
            disk_used_gb: Disk space currently in use, in GB.
            disk_total_gb: Total disk capacity, in GB.
        """
        self._call_js(
            "setSystemStats",
            f"{cpu_percent:.0f}",
            f"{ram_percent:.0f}",
            f"{ram_used_gb:.1f} / {ram_total_gb:.1f} GB",
            f"{disk_percent:.0f}",
            f"{disk_used_gb:.0f} / {disk_total_gb:.0f} GB",
        )

    def _call_js(self, function_name: str, *args: str) -> None:
        """Safely call a JS function in the window, logging failures."""
        try:
            arg_literals = ", ".join(json.dumps(a) for a in args)
            self._window.evaluate_js(f"{function_name}({arg_literals})")
        except Exception:
            log.error("chat_window_js_call_failed", function=function_name, exc_info=True)
