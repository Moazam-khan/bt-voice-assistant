"""Windows auto-start management for BT.

Uses the current user's Run registry key (HKEY_CURRENT_USER), not the
all-users key (HKEY_LOCAL_MACHINE) — this requires no admin elevation and
only affects the current user's login, not the whole machine.
"""

from __future__ import annotations

import sys
import winreg

from bt_core.logging_setup import get_logger

log = get_logger(__name__)

_RUN_KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
_VALUE_NAME = "BT"


def _launch_command() -> str:
    """Build the command Windows should run at login to start BT."""
    return f'"{sys.executable}" -m bt_core.main'


def is_enabled() -> bool:
    """Check whether BT is currently registered to auto-start.

    Returns:
        True if a BT entry exists in the Run key.
    """
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY_PATH, 0, winreg.KEY_READ) as key:
            winreg.QueryValueEx(key, _VALUE_NAME)
            return True
    except FileNotFoundError:
        return False


def enable() -> None:
    """Register BT to start automatically when the user logs into Windows."""
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY_PATH, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, _VALUE_NAME, 0, winreg.REG_SZ, _launch_command())
    log.info("autostart_enabled", command=_launch_command())


def disable() -> None:
    """Remove BT from Windows auto-start. Safe to call if not currently enabled."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY_PATH, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, _VALUE_NAME)
        log.info("autostart_disabled")
    except FileNotFoundError:
        pass
