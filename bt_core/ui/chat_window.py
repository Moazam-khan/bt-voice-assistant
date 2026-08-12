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


class ChatWindow:
    """Wraps a pywebview window showing BT's conversation transcript."""

    def __init__(self) -> None:
        """Create the window (not shown until :meth:`start` is called)."""
        html = _HTML_PATH.read_text(encoding="utf-8")
        self._window = webview.create_window(
            "BT", html=html, width=420, height=640, background_color="#0f1115"
        )

    def start(self, on_ready: Callable[[], None]) -> None:
        """Show the window and block the calling thread until it's closed.

        Must be called from the main thread.

        Args:
            on_ready: Called once the window is created and ready to
                receive JS calls — the natural place to start background
                work (e.g. the pipeline's asyncio loop on another thread).
        """
        webview.start(on_ready)

    def show_user_message(self, text: str) -> None:
        """Append a user message bubble to the transcript."""
        self._call_js("addUserMessage", text)

    def show_bt_message(self, text: str) -> None:
        """Append a BT message bubble to the transcript."""
        self._call_js("addBtMessage", text)

    def set_status(self, status: str) -> None:
        """Update the status indicator.

        Args:
            status: One of "idle", "listening", "thinking", "speaking".
        """
        self._call_js("setStatus", status)

    def _call_js(self, function_name: str, *args: str) -> None:
        """Safely call a JS function in the window, logging failures."""
        try:
            arg_literals = ", ".join(json.dumps(a) for a in args)
            self._window.evaluate_js(f"{function_name}({arg_literals})")
        except Exception:
            log.error("chat_window_js_call_failed", function=function_name, exc_info=True)
