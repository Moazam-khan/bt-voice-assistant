"""Manual test: does the chat window's Start button correctly call into Python?

Not part of bt_core -- throwaway verification script. Isolates the
click -> js_api -> Python callback wiring from the real mic-dependent
pipeline, since the known missing-microphone issue would otherwise crash
the app before the button could be tested.

Run:
    python scripts/test_start_button.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bt_core.config import get_settings
from bt_core.logging_setup import configure_logging, get_logger
from bt_core.ui.chat_window import ChatWindow

log = get_logger(__name__)


def main() -> None:
    """Show the chat window and log when Start is clicked."""
    settings = get_settings()
    configure_logging(settings.logging)

    window = ChatWindow()

    def on_start() -> None:
        log.info("start_button_clicked")
        window.set_status("listening")
        window.show_bt_message("(test) Start button worked -- I heard the click.")

    window.set_start_listening_handler(on_start)
    window.start(on_ready=lambda: log.info("window_ready", message="click the Start button"))


if __name__ == "__main__":
    main()
