"""System tray icon for BT.

pystray's Icon.run() blocks its calling thread with a native Windows
message loop, so it runs on a dedicated background thread rather than the
asyncio event loop. The tray thread and the asyncio loop communicate only
through the on_quit callback, which must itself be thread-safe (the
caller is expected to schedule any asyncio-side work via
loop.call_soon_threadsafe, since this module has no asyncio awareness).
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import pystray
from PIL import Image, ImageDraw

from bt_core import autostart
from bt_core.logging_setup import get_logger

log = get_logger(__name__)

_ICON_SIZE = 64
_ICON_COLOR = (37, 99, 235, 255)


def _build_icon_image() -> Image.Image:
    """Draw a simple filled-circle placeholder icon for BT."""
    image = Image.new("RGBA", (_ICON_SIZE, _ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = 4
    draw.ellipse((margin, margin, _ICON_SIZE - margin, _ICON_SIZE - margin), fill=_ICON_COLOR)
    return image


class TrayIcon:
    """Wraps a pystray icon with a status label, auto-start toggle, and quit."""

    def __init__(self, on_quit: Callable[[], None]) -> None:
        """Initialize the tray icon.

        Args:
            on_quit: Called (on the tray's background thread) when the
                user selects "Quit" from the menu.
        """
        self._on_quit = on_quit
        menu = pystray.Menu(
            pystray.MenuItem("BT is running", None, enabled=False),
            pystray.MenuItem(
                "Start with Windows",
                self._toggle_autostart,
                checked=lambda _: autostart.is_enabled(),
            ),
            pystray.MenuItem("Quit", self._handle_quit),
        )
        self._icon = pystray.Icon("bt", _build_icon_image(), "BT Voice Assistant", menu)
        self._thread: threading.Thread | None = None

    def _toggle_autostart(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        """Flip BT's Windows auto-start registration on/off."""
        if autostart.is_enabled():
            autostart.disable()
        else:
            autostart.enable()

    def _handle_quit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        log.info("tray_quit_selected")
        icon.stop()
        self._on_quit()

    def start(self) -> None:
        """Start the tray icon on a background thread."""
        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()
        log.info("tray_icon_started")

    def stop(self) -> None:
        """Stop the tray icon."""
        self._icon.stop()
